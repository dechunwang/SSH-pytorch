import argparse

import os
import numpy as np
from model.dataset.factory import get_imdb
from model.utils.config import cfg
from model.roi_data_layer.layer import RoIDataLayer
import torch
from model.SSH import SSH
from model.network import save_check_point, load_check_point
import cv2
from model.utils.timer import Timer
import torch.optim as optim
from model.utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections

def parser():
    parser = argparse.ArgumentParser('SSH Train module')
    parser.add_argument('--gpu_ids', dest='gpu_ids', default='0', type=str,
                        help='gpu devices  to be used')
    parser.add_argument('--model_path', dest='model_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')
    parser.add_argument('--model_save_path', dest='model_save_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')
    parser.add_argument('--max_iters', dest='max_iters', default=450000, type=int,
                        help='maximum iterations')

    return parser.parse_args()


def get_training_roidb(imdb):
    """
    Get the training roidb given an imdb
    :param imdb: The training imdb
    :return: The training roidb
    """

    def filter_roidb(roidb):
        """
        Filtering samples without positive and negative training anchors
        :param roidb: the training roidb
        :return: the filtered roidb
        """

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LOW))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        return filtered_roidb

    # Augment imdb with flipped images
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    # Add required information to imdb
    imdb.prepare_roidb()
    # Filter the roidb
    final_roidb = filter_roidb(imdb.roidb)
    print('done')
    return final_roidb


def train(net, optimizer, imdb, roidb, arg):
    max_iters = arg.max_iters
    iter = 1
    display_interval = cfg.TRAIN.DISPLAY
    train_data = RoIDataLayer(roidb, imdb.num_classes)

    loss_sum = 0
    m3_ssh_cls_loss_sum = 0
    m3_bbox_loss_sum = 0
    m2_ssh_cls_loss_sum = 0
    m2_bbox_loss_sum = 0
    m1_ssh_cls_loss_sum = 0
    m1_bbox_loss_sum = 0
    timer = {"forward": Timer(), "data": Timer()}

    im = cv2.imread("/home/dwang/SynologyDrive/pyt_example/data/datasets/wider/WIDER_train/images/28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_39.jpg")



    im_scale = _compute_scaling_factor(im.shape, cfg.TRAIN.SCALES[0], cfg.TRAIN.MAX_SIZE)
    bbox = [122, 2, 752, 688]
    bbox = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3],1]], np.float)
    bbox *=im_scale
    bbox [:,4]=1
    im_blob = _get_image_blob(im, [im_scale])[0]

    im_info = np.array([[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
    im_data = im_blob['data']

    im_data = torch.from_numpy(im_data).to(device)
    # add a batch dimension
    im_info = torch.from_numpy(im_info).to(device)
    gt_boxes = torch.from_numpy(bbox).to(device).unsqueeze(0).float()


    # img = np.squeeze(blobs['data'])
    #
    # img=img.transpose(1,2,0)
    # for i in range(len(blobs['gt_boxes'])):
    #     pt1 = tuple(blobs['gt_boxes'][i, 0:2])
    #     pt2 = tuple(blobs['gt_boxes'][i, 2:4])
    #     cv2.rectangle(img, pt1, pt2, (255, 255, 255))
    # cv2.imwrite("train.jpg", img)

    optimizer.zero_grad()



    m3_ssh_cls_loss, m2_ssh_cls_loss, m1_ssh_cls_loss, \
    m3_bbox_loss, m2_bbox_loss, m1_bbox_loss = net(im_data, im_info, gt_boxes)

    loss = (m3_ssh_cls_loss + m2_ssh_cls_loss + m1_ssh_cls_loss + \
            m3_bbox_loss + m2_bbox_loss + m1_bbox_loss)

    m3_ssh_cls_loss_sum += m3_ssh_cls_loss.item()
    m3_bbox_loss_sum += m3_bbox_loss.item()
    m2_ssh_cls_loss_sum += m2_ssh_cls_loss.item()
    m2_bbox_loss_sum += m2_bbox_loss.item()
    m1_ssh_cls_loss_sum += m1_ssh_cls_loss.item()
    m1_bbox_loss_sum += m1_bbox_loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(),0.5)
    optimizer.step()

    loss_sum += loss.item()

    timer["forward"].toc()

    # if m3_bbox_loss.item() == 0 :
    #     img = np.squeeze(blobs['data'])
    #
    #     img=img.transpose(1,2,0)
    #     for i in range(len(blobs['gt_boxes'])):
    #         pt1 = tuple(blobs['gt_boxes'][i, 0:2])
    #         pt2 = tuple(blobs['gt_boxes'][i, 2:4])
    #         cv2.rectangle(img, pt1, pt2, (255, 255, 255))
    #     cv2.imwrite("zero/loss_0_{}.jpg".format(iter), img)
    #     f = open("zero/loss_0_{}.txt".format(iter), "a")
    #     f.write(blobs['file_path'])


    if (iter % display_interval == 0):
        loss_average = loss_sum / display_interval
        m3_ssh_cls_loss_average = m3_ssh_cls_loss_sum / display_interval
        m3_bbox_loss_average = m3_bbox_loss_sum / display_interval
        m2_ssh_cls_loss_average = m2_ssh_cls_loss_sum / display_interval
        m2_bbox_loss_average = m2_bbox_loss_sum / display_interval
        m1_ssh_cls_loss_average = m1_ssh_cls_loss_sum / display_interval
        m1_bbox_loss_average = m1_bbox_loss_sum / display_interval

        loss_sum = 0
        m3_ssh_cls_loss_sum = 0
        m3_bbox_loss_sum = 0
        m2_ssh_cls_loss_sum = 0
        m2_bbox_loss_sum = 0
        m1_ssh_cls_loss_sum = 0
        m1_bbox_loss_sum = 0

        print("------------------------iteration {}-----------{} left---------".format(iter, max_iters - iter))
        print("Average per iter: {:.4f} second.   ETA: {:.4f} hours".format(timer["forward"].average_time,
                                                                            (max_iters - iter) * (
                                                                                timer["forward"].average_time) / (
                                                                                        60 * 60)))
        print("Average data load time: {:.4f}".format(timer["data"].average_time))
        print('loss:{}\nm3 cls:{}\nm3 box:{}\nm2 cls:{}\nm2 box:{}'
              '\nm1 cls:{}\nm1 box:{} '.format(loss_average, m3_ssh_cls_loss_average, m3_bbox_loss_average,
                                               m2_ssh_cls_loss_average, m2_bbox_loss_average,
                                               m1_ssh_cls_loss_average, m1_bbox_loss_average))
        timer["forward"].reset()
        timer["data"].reset()

    if iter % cfg.TRAIN.CHECKPOINT == 0:
        save_check_point(arg.model_save_path, iter, loss, net, optimizer)
        print("check point saved")


if __name__ == '__main__':
    arg = parser()
    vgg16_image_net = True
    if (os.path.isfile(arg.model_path)):
        vgg16_image_net = False

    imdb = get_imdb('wider_train')
    roidb = get_training_roidb(imdb)

    assert len(str(arg.gpu_ids)) == 1, "only single gpu is supported, " \
                                       "use train_dist.py for multiple gpu support"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = SSH(vgg16_image_net)
    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if not vgg16_image_net:
        check_point = load_check_point(arg.model_path)
        net.load_state_dict(check_point['model_state_dict'])
        # iter = check_point['iteration']
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        # for param_tensor in net.state_dict():
        #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])
    net.to(device)
    net.train()

    train(net, optimizer, imdb, roidb, arg)
