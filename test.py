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
import torch.optim as optim

from model.nms.nms_wrapper import nms
from model.utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections


def parser():
    parser = argparse.ArgumentParser('SSH Train module')
    parser.add_argument('--img', dest='img', default='demo/demo.jpg', type=str,
                        help='image to be test')
    parser.add_argument('--img_out', dest='img_out', default='demo/demo_result.jpg',
                        help='visialized image output')
    parser.add_argument('--gpu_ids', dest='gpu_ids', default='0', type=str,
                        help='gpu devices  to be used')
    parser.add_argument('--thresh', dest='thresh', default=0.5, type=float,
                        help='Detections with a probability less than this threshold are ignored')
    parser.add_argument('--model_path', dest='model_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')

    return parser.parse_args()


if __name__ == '__main__':
    arg = parser()
    filepath = arg.img
    output_path = os.path.dirname(arg.img_out)
    output_name = os.path.basename(arg.img_out)
    visualize = True

    thresh = arg.thresh
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    saved_model_path = arg.model_path
    assert os.path.isfile(saved_model_path), 'Pretrained model not found'

    net = SSH(vgg16_image_net=False)

    if (os.path.isfile(saved_model_path)):
        check_point = load_check_point(saved_model_path)
        net.load_state_dict(check_point['model_state_dict'])
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    net.to(device)
    net.eval()

    with torch.no_grad():

        im = cv2.imread(filepath)
        im_scale = _compute_scaling_factor(im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])[0]

        im_info = np.array([[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
        im_data = im_blob['data']

        im_info = torch.from_numpy(im_info).to(device)
        im_data = torch.from_numpy(im_data).to(device)

        batch_size = im_data.size()[0]
        ssh_rois = net(im_data, im_info)

        inds = (ssh_rois[:, :, 4] > thresh)
        # inds=inds.unsqueeze(2).expand(batch_size,inds.size()[1],5)
        #
        # ssh_roi_keep = ssh_rois[inds].view(batch_size,-1,5)
        ssh_roi_keep = ssh_rois[:, inds[0], :]
        # unscale back
        ssh_roi_keep[:, :, 0:4] /= im_scale

        for i in range(batch_size):
            ssh_roi_single = ssh_roi_keep[i].cpu().numpy()
            nms_keep = nms(ssh_roi_single, cfg.TEST.RPN_NMS_THRESH)
            cls_dets_single = ssh_roi_single[nms_keep, :]
            if visualize:
                visusalize_detections(im, cls_dets_single, plt_name=output_name,
                                      visualization_folder=output_path)

        print(cls_dets_single)

        print(cls_dets_single.shape)
