import cv2
import torch
import os, sys,argparse
from model.SSH import SSH
from model.utils.config import cfg, get_output_dir
from model.nms.nms_wrapper import nms
from model.utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections
from model.utils.timer import Timer
from model.network import load_check_point
from model.dataset.factory import get_imdb
import numpy as np

def parser():
    parser = argparse.ArgumentParser('SSH Train module')
    parser.add_argument('--db', dest='db_name', help='Path to the image',
                        default='wider_val', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='output', type=str)
    parser.add_argument('--gpu_ids', dest='gpu_ids', default='0', type=str,
                        help='gpu devices  to be used')
    parser.add_argument('--thresh', dest='thresh', default=0.05, type=float,
                        help='Detections with a probability less than this threshold are ignored')
    parser.add_argument('--vis', dest='visualize', help='visualize detections',type=bool,default=False)
    parser.add_argument('--vis_path', dest='visualize_folder', help='visualize result folder', type=str, default="output/result")
    parser.add_argument('--model_path', dest='model_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')

    return parser.parse_args()


def forward(net, im_blob,im_scale,device,thresh=0.5):
    im_info = np.array([[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
    im_data = im_blob['data']


    with torch.no_grad():
        im_info = torch.from_numpy(im_info).to(device)
        im_data = torch.from_numpy(im_data).to(device)
        ssh_rois = net(im_data, im_info)
        # Detections with a probability less than this threshold are ignored
        inds = (ssh_rois[:, :, 4] > thresh)
        ssh_roi_keep = ssh_rois[:, inds[0], :]

        # unscale back
        if ssh_roi_keep.dim()==1:
            ssh_roi_keep= ssh_rois

        ssh_roi_keep[:, :, 0:4] /= im_scale

        ssh_roi_single = ssh_roi_keep[0]


    return ssh_roi_single.cpu().numpy()

def detect(net, im_path,device, thresh=0.5, visualize=False, timers=None, pyramid=False, visualization_folder=None):
    """
    Main module to detect faces
    :param net: The trained network
    :param im_path: The path to the image
    :param device: GPU or CPU device to be used
    :param thresh: Detection with a less score than thresh are ignored
    :param visualize: Whether to visualize the detections
    :param timers: Timers for calculating detect time (if None new timers would be created)
    :param pyramid: Whether to use pyramid during inference
    :param visualization_folder: If set the visualizations would be saved in this folder (if visualize=True)
    :return: cls_dets (bounding boxes concatenated with scores) and the timers
    """

    if not timers:
        timers = {'detect': Timer(),
                  'misc': Timer()}

    im = cv2.imread(im_path)
    imfname = os.path.basename(im_path)
    sys.stdout.flush()
    timers['detect'].tic()

    if not pyramid:
        im_scale = _compute_scaling_factor(im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])[0]
        ssh_rois = forward(net,im_blob,im_scale,device,thresh)

    else :
        assert False, 'not implement'

    timers['detect'].toc()
    timers['misc'].tic()

    nms_keep = nms(ssh_rois, cfg.TEST.RPN_NMS_THRESH)
    cls_dets = ssh_rois[nms_keep, :]

    if visualize:
        plt_name = os.path.splitext(imfname)[0] + '_detections_{}'.format("SSH pytorch")
        visusalize_detections(im, cls_dets, plt_name=plt_name, visualization_folder=visualization_folder)
    timers['misc'].toc()
    return cls_dets, timers




def test_net(net, imdb, device,thresh=0.5, visualize=False,output_path=None):
    """
       Testing the SSH network on a dataset
       :param net: The trained network
       :param imdb: The test imdb
       :param thresh: Detections with a probability less than this threshold are ignored
       :param visualize: Whether to visualize the detections
       :param output_path: Output directory
    """

    print('Evaluating on {}'.format(imdb.name))
    output_dir = get_output_dir(imdb_name=imdb.name, net_name="SSH pytorch", output_dir=output_path)

    timers = {'detect': Timer(), 'misc': Timer()}
    dets = [[[] for _ in range(len(imdb))] for _ in range(imdb.num_classes)]

    pyramid = True if len(cfg.TEST.SCALES) > 1 else False

    for i in range(len(imdb)):
        im_path = imdb.image_path_at(i)
        dets[1][i], detect_time = detect(net, im_path, device, thresh, visualize=visualize,
                                         timers=timers, pyramid=pyramid,visualization_folder=arg.visualize_folder)
        print('\r{:d}/{:d} detect-time: {:.3f}s, misc-time:{:.3f}s'
              .format(i + 1, len(imdb), timers['detect'].average_time,
                      timers['misc'].average_time), end='')

    print('\n', end='')

    # Evaluate the detections
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes=dets, output_dir=output_dir, method_name="SSH pytorch")
    print('All Done!')


if __name__ == '__main__':
    arg = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    saved_model_path = arg.model_path
    assert os.path.isfile(saved_model_path), 'Pretrained model not found'

    imdb = get_imdb(arg.db_name)

    net = SSH(vgg16_image_net=False)

    if (os.path.isfile(saved_model_path)):
        check_point = load_check_point(saved_model_path)
        net.load_state_dict(check_point['model_state_dict'])
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    net.to(device)
    net.eval()
    # Evaluate the network
    test_net(net, imdb, device, arg.thresh,visualize=arg.visualize, output_path=arg.out_path)
