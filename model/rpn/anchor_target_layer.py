import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from model.utils.bbox import bbox_overlaps_batch , bbox_transform_batch
from  model.utils.config import cfg
from model.rpn.generate_anchors import generate_anchors


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios , allowed_border, name):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        self._name=name
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = allowed_border  # default is 0

    def forward(self, rpn_cls_score,gt_boxes,im_info,rpn_cls_score_OHEM = None):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        # rpn_cls_score = input[0]
        # gt_boxes = input[1]
        # im_info = input[2]
        # num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes)  # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) & # width
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))  # height

        inds_inside = torch.nonzero(keep).view(-1)

        # if inds_inside.size()[0] == 0 :
        #     inds_inside = torch.arange(0,inds_inside.size()[0])
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care


        target_size= inds_inside.size(0)

        labels = gt_boxes.new_full((batch_size, target_size),fill_value=-1)
        bbox_inside_weights = gt_boxes.new_full((batch_size, target_size),fill_value=0)
        bbox_outside_weights = gt_boxes.new_full((batch_size, target_size),fill_value=0)

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)



        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP] = 1

        # if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.ANCHOR_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)



        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                if cfg.TRAIN.HARD_POSITIVE_MINING and rpn_cls_score_OHEM is not None:
                    ohem_scores = rpn_cls_score_OHEM[i, self._num_anchors:, :, :]
                    # ohem_score (A,H,W) to (H,W,A)
                    ohem_scores = ohem_scores.permute(1, 2, 0).contiguous()
                    # ohem_score (H*W*A)
                    ohem_scores = ohem_scores.view(-1,1)
                    ohem_scores = ohem_scores[inds_inside]
                    # find lowest predicted score
                    pos_ohem_scores = 1 - ohem_scores[fg_inds]
                    #sort by descending order
                    _, orderd_ohem_score = torch.sort(pos_ohem_scores,dim = 0,descending = True)
                    # sample ohem score
                    ohem_sampled_fgs = fg_inds[orderd_ohem_score[:num_fg]]
                    labels[i][fg_inds] = -1
                    labels[i][ohem_sampled_fgs] = 1
                else:
                    rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                    labels[i][disable_inds] = -1

            #           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()
                if cfg.TRAIN.HARD_NEGATIVE_MINING and rpn_cls_score_OHEM is not None:
                    ohem_scores = rpn_cls_score_OHEM[i, self._num_anchors:, :, :]
                    # ohem_score (A,H,W) to (H,W,A)
                    ohem_scores = ohem_scores.permute(1, 2, 0).contiguous()
                    # ohem_score (H*W*A)
                    ohem_scores = ohem_scores.view(-1,1)
                    ohem_scores = ohem_scores[inds_inside]
                    # find Highest predicted score
                    neg_ohem_scores = ohem_scores[bg_inds]
                    # sort by descending order
                    _, orderd_ohem_score = torch.sort(neg_ohem_scores, dim = 0, descending=True)
                    # sample ohem score
                    ohem_sampled_bgs = bg_inds[orderd_ohem_score[:num_bg]]
                    labels[i][bg_inds] = -1
                    labels[i][ohem_sampled_bgs] = 0
                else:
                    rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                    labels[i][disable_inds] = -1

        # sum_fg = torch.sum((labels == 1).int(), 1)
        # sum_bg = torch.sum((labels == 0).int(), 1)
        # print("name={}, fg={}, bg={}".format(self._name,sum_fg[0],sum_bg[0]))

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors,
                                              gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[0] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        # outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()
        # outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                            4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        # outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                              4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()
        # outputs.append(bbox_outside_weights)

        # return outputs
        return labels , bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
