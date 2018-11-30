import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, same_padding=False, stride=1, relu=True, bn=False):
        super(Conv2D, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class M3(nn.Module):
    def __init__(self, in_channels):
        super(M3, self).__init__()
        self.m3_ssh_3x3 = Conv2D(in_channels, 256, 3, True, 1, False)
        self.m3_ssh_dimred = Conv2D(in_channels, 128, 3, True, 1, True)
        self.m3_ssh_5x5 = Conv2D(128, 128, 3, True, 1, False)
        self.m3_ssh_7x7_1 = Conv2D(128, 128, 3, True, 1, True)
        self.m3_ssh_7x7 = Conv2D(128, 128, 3, True, 1, False, )
        self.m3_ssh_cls_score = Conv2D(128 * 2 + 256, 4, 1, False, 1, False)
        self.m3_ssh_bbox_pred = Conv2D(128 * 2 + 256, 8, 1, False, 1, False)

    def forward(self, pool6):
        m3_ssh_3x3 = self.m3_ssh_3x3(pool6)
        m3_ssh_dimred = self.m3_ssh_dimred(pool6)
        m3_ssh_5x5 = self.m3_ssh_5x5(m3_ssh_dimred)
        m3_ssh_7x7_1 = self.m3_ssh_7x7_1(m3_ssh_dimred)
        m3_ssh_7x7 = self.m3_ssh_7x7(m3_ssh_7x7_1)
        m3_ssh_output = F.relu(torch.cat((m3_ssh_3x3, m3_ssh_5x5, m3_ssh_7x7), dim=1))
        m3_ssh_cls_score = self.m3_ssh_cls_score(m3_ssh_output)
        m3_ssh_bbox_pred = self.m3_ssh_bbox_pred(m3_ssh_output)

        return m3_ssh_cls_score, m3_ssh_bbox_pred


class M2(nn.Module):
    def __init__(self, in_channels):
        super(M2, self).__init__()
        self.m2_ssh_3x3 = Conv2D(in_channels, 256, 3, True, 1, False)
        self.m2_ssh_dimred = Conv2D(in_channels, 128, 3, True, 1, True)
        self.m2_ssh_5x5 = Conv2D(128, 128, 3, True, 1, False)
        self.m2_ssh_7x7_1 = Conv2D(128, 128, 3, True, 1, True)
        self.m2_ssh_7x7 = Conv2D(128, 128, 3, True, 1, False, )
        self.m2_ssh_cls_score = Conv2D(128 * 2 + 256, 4, 1, False, 1, False)
        self.m2_ssh_bbox_pred = Conv2D(128 * 2 + 256, 8, 1, False, 1, False)

    def forward(self, conv5_3):
        m2_ssh_dimred = self.m2_ssh_dimred(conv5_3)
        m2_ssh_3x3 = self.m2_ssh_3x3(conv5_3)
        m2_ssh_5x5 = self.m2_ssh_5x5(m2_ssh_dimred)
        m2_ssh_7x7_1 = self.m2_ssh_7x7_1(m2_ssh_dimred)
        m2_ssh_7x7 = self.m2_ssh_7x7(m2_ssh_7x7_1)
        m2_ssh_output = F.relu(torch.cat((m2_ssh_3x3, m2_ssh_5x5, m2_ssh_7x7), dim=1))
        m2_ssh_cls_score = self.m2_ssh_cls_score(m2_ssh_output)
        m2_ssh_bbox_pred = self.m2_ssh_bbox_pred(m2_ssh_output)

        return m2_ssh_cls_score, m2_ssh_bbox_pred


class M1(nn.Module):
    def __init__(self, in_channels):
        super(M1, self).__init__()
        self.m1_ssh_3x3 = Conv2D(in_channels, 128, 3, True, 1, False)
        self.m1_ssh_dimred = Conv2D(in_channels, 64, 3, True, 1, True)
        self.m1_ssh_5x5 = Conv2D(64, 64, 3, True, 1, False)
        self.m1_ssh_7x7_1 = Conv2D(64, 64, 3, True, 1, True)
        self.m1_ssh_7x7 = Conv2D(64, 64, 3, True, 1, False, )
        self.m1_ssh_cls_score = Conv2D(64 * 2 + 128, 4, 1, False, 1, False)
        self.m1_ssh_bbox_pred = Conv2D(64 * 2 + 128, 8, 1, False, 1, False)

    def forward(self, conv4_fuse_final):
        m1_ssh_dimred = self.m1_ssh_dimred(conv4_fuse_final)
        m1_ssh_3x3 = self.m1_ssh_3x3(conv4_fuse_final)
        m1_ssh_5x5 = self.m1_ssh_5x5(m1_ssh_dimred)
        m1_ssh_7x7_1 = self.m1_ssh_7x7_1(m1_ssh_dimred)
        m1_ssh_7x7 = self.m1_ssh_7x7(m1_ssh_7x7_1)
        m1_ssh_output = F.relu(torch.cat((m1_ssh_3x3, m1_ssh_5x5, m1_ssh_7x7), dim=1))
        m1_ssh_cls_score = self.m1_ssh_cls_score(m1_ssh_output)
        m1_ssh_bbox_pred = self.m1_ssh_bbox_pred(m1_ssh_output)

        return m1_ssh_cls_score, m1_ssh_bbox_pred


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def save_check_point(path,iteration,loss,net,optimizer):
    torch.save({
        'iteration': iteration,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, path)
def load_check_point(path):
    return torch.load(path)