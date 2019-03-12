import torch
import model.network as net
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
import math
from torch.autograd import Variable
import torchvision as tv
from model.rpn.anchor_target_layer import _AnchorTargetLayer
from model.rpn.proposal_layer import _Proposallayer
import numpy as np


class SSH(nn.Module):
    def __init__(self, vgg16_image_net=True):
        super(SSH, self).__init__()

        vgg16_features = tv.models.vgg16(pretrained=vgg16_image_net).features
        self.conv4_3 = nn.Sequential(*list(vgg16_features.children())[:23])
        self.conv5_3 = nn.Sequential(*list(vgg16_features.children())[23:30])
        self.m3_module = net.M3(512)
        self.m2_module = net.M2(512)
        self.m1_module = net.M1(128)

        self.conv5_128 = net.Conv2D(512, 128, 1, False, 1, True)
        self.conv5_128_up = nn.ConvTranspose2d(128, 128, 4, 2, 1, 1, 128, False)
        self.con4_128 = net.Conv2D(512, 128, 1, False, 1, True)
        self.con4_fuse_final = net.Conv2D(128, 128, 3, True, 1, True)

        self.pool6 = nn.MaxPool2d(2, 2)

        self.m3_anchor_target_layer = _AnchorTargetLayer(32, np.array([16, 32]), np.array([1, ]), 512, name='m3')
        self.m2_anchor_target_layer = _AnchorTargetLayer(16, np.array([4, 8]), np.array([1, ]), 0, name='m2')
        self.m1_anchor_target_layer = _AnchorTargetLayer(8, np.array([1, 2]), np.array([1, ]), 0, name='m1')

        self.m3_proposal_layer = _Proposallayer(32, np.array([16, 32]), np.array([1, ]))
        self.m2_proposal_layer = _Proposallayer(16, np.array([4, 8]), np.array([1, ]))
        self.m1_proposal_layer = _Proposallayer(8, np.array([1, 2]), np.array([1, ]))

        self.m3_soft_max = nn.Softmax(1)
        self.m2_soft_max = nn.Softmax(1)
        self.m1_soft_max = nn.Softmax(1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, image_data, im_info, gt_boxes=None):
        batch_size = image_data.size(0)

        conv4_3 = self.conv4_3(image_data)
        conv5_3 = self.conv5_3(conv4_3)

        m2_ssh_cls_score, m2_ssh_bbox_pred = self.m2_module(conv5_3)

        # M3
        pool6 = self.pool6(conv5_3)
        m3_ssh_cls_score, m3_ssh_bbox_pred = self.m3_module(pool6)

        # M 1
        conv4_128 = self.con4_128(conv4_3)
        conv5_128 = self.conv5_128(conv5_3)
        conv5_128_up = self.conv5_128_up(conv5_128)

        # crop cove5_128_up to match conv4_128's size
        # NCHW
        conv4_128_height = conv4_128.size()[2]
        conv4_128_width = conv4_128.size()[3]

        conv5_128_crop = conv5_128_up[:, :,
                         0:conv4_128_height,
                         0:conv4_128_width]

        conv4_fuse = conv5_128_crop + conv4_128

        con4_fuse_final = self.con4_fuse_final(conv4_fuse)
        m1_ssh_cls_score, m1_ssh_bbox_pred = self.m1_module(con4_fuse_final)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            m3_ssh_cls_prob_reshape_OHEM = None
            m2_ssh_cls_prob_reshape_OHEM = None
            m1_ssh_cls_prob_reshape_OHEM = None

            if cfg.TRAIN.HARD_POSITIVE_MINING or cfg.TRAIN.HARD_NEGATIVE_MINING:
                m3_ssh_cls_score_reshape_OHEM = self.reshape(m3_ssh_cls_score.detach(), 2)
                m2_ssh_cls_score_reshape_OHEM = self.reshape(m2_ssh_cls_score.detach(), 2)
                m1_ssh_cls_score_reshape_OHEM = self.reshape(m1_ssh_cls_score.detach(), 2)

                # softmax
                m3_ssh_cls_prob_output_OHEM = self.m3_soft_max(m3_ssh_cls_score_reshape_OHEM)
                m2_ssh_cls_prob_output_OHEM = self.m2_soft_max(m2_ssh_cls_score_reshape_OHEM)
                m1_ssh_cls_prob_output_OHEM = self.m1_soft_max(m1_ssh_cls_score_reshape_OHEM)

                # reshape from (batch,2,2*H,W) back to (batch,4,h,w)
                m3_ssh_cls_prob_reshape_OHEM = self.reshape(m3_ssh_cls_prob_output_OHEM, 4)
                m2_ssh_cls_prob_reshape_OHEM = self.reshape(m2_ssh_cls_prob_output_OHEM, 4)
                m1_ssh_cls_prob_reshape_OHEM = self.reshape(m1_ssh_cls_prob_output_OHEM, 4)

            m3_labels, m3_bbox_targets, m3_bbox_inside_weights, m3_bbox_outside_weights = \
                self.m3_anchor_target_layer(m3_ssh_cls_score, gt_boxes, im_info, m3_ssh_cls_prob_reshape_OHEM)

            m2_labels, m2_bbox_targets, m2_bbox_inside_weights, m2_bbox_outside_weights = \
                self.m2_anchor_target_layer(m2_ssh_cls_score, gt_boxes, im_info, m2_ssh_cls_prob_reshape_OHEM)

            m1_labels, m1_bbox_targets, m1_bbox_inside_weights, m1_bbox_outside_weights = \
                self.m1_anchor_target_layer(m1_ssh_cls_score, gt_boxes, im_info, m1_ssh_cls_prob_reshape_OHEM)

            # reshape from (batch,4,h,w) to (batch,2,2*h,w)
            m3_ssh_cls_score_reshape = self.reshape(m3_ssh_cls_score, 2)
            m2_ssh_cls_score_reshape = self.reshape(m2_ssh_cls_score, 2)
            m1_ssh_cls_score_reshape = self.reshape(m1_ssh_cls_score, 2)

            # reshape from (batch, 2, 2*h,w) to (batch, 2*h*w,2)
            m3_ssh_cls_score = m3_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            m2_ssh_cls_score = m2_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            m1_ssh_cls_score = m1_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

            # reshape from (batch,1,2*H,W) to (batch, 2*h*w)
            m3_target_labels = m3_labels.view(batch_size, -1)
            m2_target_labels = m2_labels.view(batch_size, -1)
            m1_target_labels = m1_labels.view(batch_size, -1)

            # reshape to (N,C) for cross_entropy loss
            m3_ssh_cls_score = m3_ssh_cls_score.view(-1, 2)
            m2_ssh_cls_score = m2_ssh_cls_score.view(-1, 2)
            m1_ssh_cls_score = m1_ssh_cls_score.view(-1, 2)

            # reshape to (N)
            m3_target_labels = m3_target_labels.view(-1).long()
            m2_target_labels = m2_target_labels.view(-1).long()
            m1_target_labels = m1_target_labels.view(-1).long()

            # compute bbox classification loss
            m3_ssh_cls_loss = F.cross_entropy(m3_ssh_cls_score, m3_target_labels, ignore_index=-1)
            m2_ssh_cls_loss = F.cross_entropy(m2_ssh_cls_score, m2_target_labels, ignore_index=-1)
            m1_ssh_cls_loss = F.cross_entropy(m1_ssh_cls_score, m1_target_labels, ignore_index=-1)

            # compute bbox regression loss
            m3_bbox_loss = net._smooth_l1_loss(m3_ssh_bbox_pred, m3_bbox_targets,
                                               m3_bbox_inside_weights, m3_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
            m2_bbox_loss = net._smooth_l1_loss(m2_ssh_bbox_pred, m2_bbox_targets,
                                               m2_bbox_inside_weights, m2_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
            m1_bbox_loss = net._smooth_l1_loss(m1_ssh_bbox_pred, m1_bbox_targets,
                                               m1_bbox_inside_weights, m1_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

            return m3_ssh_cls_loss, m2_ssh_cls_loss, m1_ssh_cls_loss, m3_bbox_loss, m2_bbox_loss, m1_bbox_loss
        else:
            # reshape from (batch,4,h,w) to (batch,2,-1,w)
            m3_ssh_cls_score_reshape = self.reshape(m3_ssh_cls_score, 2)
            m2_ssh_cls_score_reshape = self.reshape(m2_ssh_cls_score, 2)
            m1_ssh_cls_score_reshape = self.reshape(m1_ssh_cls_score, 2)

            # softmax
            m3_ssh_cls_prob_output = self.m3_soft_max(m3_ssh_cls_score_reshape)
            m2_ssh_cls_prob_output = self.m2_soft_max(m2_ssh_cls_score_reshape)
            m1_ssh_cls_prob_output = self.m1_soft_max(m1_ssh_cls_score_reshape)

            # reshape from (batch,2,2*H,W) back to (batch,4,h,w)
            m3_ssh_cls_prob_reshape = self.reshape(m3_ssh_cls_prob_output, 4)
            m2_ssh_cls_prob_reshape = self.reshape(m2_ssh_cls_prob_output, 4)
            m1_ssh_cls_prob_reshape = self.reshape(m1_ssh_cls_prob_output, 4)

            # roi has shape of (batch, top_k, 5)
            # where (batch, top_k, 4) is cls score and
            # (batch, top_k, 0:4) is bbox coordinated
            m3_ssh_roi = self.m3_proposal_layer(m3_ssh_cls_prob_reshape, m3_ssh_bbox_pred, im_info)
            m2_ssh_roi = self.m2_proposal_layer(m2_ssh_cls_prob_reshape, m2_ssh_bbox_pred, im_info)
            m1_ssh_roi = self.m1_proposal_layer(m1_ssh_cls_prob_reshape, m1_ssh_bbox_pred, im_info)

            ssh_roi = torch.cat((m3_ssh_roi, m2_ssh_roi, m1_ssh_roi), dim=1)
            # ssh_roi = torch.cat((m3_ssh_roi,), dim=1)
            return ssh_roi
