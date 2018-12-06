from train import get_training_roidb
import torch
import torch.distributed as dist
import torch.optim as optim
import argparse
from model.SSH import SSH
import cv2
import numpy as np
from model.dataset.factory import get_imdb
from model.utils.config import cfg
from model.roi_data_layer.layer import RoIDataLayer
from model.network import load_check_point, save_check_point
from model.utils.timer import Timer
from torch.multiprocessing import Process
import os


def parser():
    parser = argparse.ArgumentParser('SSH Train module')
    parser.add_argument('--gpu', dest='gpu_ids', default='0', type=str,
                        help='Multi gpu devices ids to be used')
    parser.add_argument('--model_path', dest='model_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')
    parser.add_argument('--model_save_path', dest='model_save_path', default='check_point/check_point.zip', type=str,
                        help='Saved model path')
    parser.add_argument('--max_iters', dest='max_iters', default=300000, type=int,
                        help='maximum iterations')
    parser.add_argument('--master_ip', dest='master_ip', default='127.0.0.1',
                        help='masters ip')
    parser.add_argument('--master_port', dest='master_port', default='29500',
                        help='masters port')
    parser.add_argument('--rank', dest='rank', default='0',
                        help='rank')
    parser.add_argument('--world_size', dest='world_size', default='0',
                        help='rank')

    return parser.parse_args()


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
        param.grad.data /= size


def run(rank, gpu_id, arg, cfg):
    imdb = get_imdb('wider_train')
    roidb = get_training_roidb(imdb)
    max_iters = int(arg.max_iters)
    vgg16_image_net = True
    if os.path.isfile(arg.model_path):
        vgg16_image_net = False
    net = SSH(vgg16_image_net)
    device = torch.device("cuda:0")
    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if not vgg16_image_net:
        print("load model from {}".format(arg.model_path))
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
    # net = torch.nn.parallel.DistributedDataParallel(net)
    net.train()

    train_data = RoIDataLayer(roidb, imdb.num_classes)
    train_data._shuffle_roidb_inds()
    iter = 1
    display_interval = cfg.TRAIN.DISPLAY

    loss_sum = 0
    m3_ssh_cls_loss_sum = 0
    m3_bbox_loss_sum = 0
    m2_ssh_cls_loss_sum = 0
    m2_bbox_loss_sum = 0
    m1_ssh_cls_loss_sum = 0
    m1_bbox_loss_sum = 0
    timer = {"forward": Timer(), "data": Timer()}

    for iter in range(iter, max_iters):
        timer["forward"].tic()
        timer["data"].tic()
        blobs = train_data.forward()
        timer["data"].toc()

        if iter == 1:
            img = np.squeeze(blobs['data'])

            img = img.transpose(1, 2, 0)
            for i in range(len(blobs['gt_boxes'])):
                pt1 = tuple(blobs['gt_boxes'][i, 0:2])
                pt2 = tuple(blobs['gt_boxes'][i, 2:4])
                cv2.rectangle(img, pt1, pt2, (255, 255, 255))
            cv2.imwrite("rank_{}_.jpg".format(rank), img)

        optimizer.zero_grad()

        im_data = torch.from_numpy(blobs['data']).to(device)
        # add a batch dimension
        im_info = torch.from_numpy(blobs['im_info']).to(device)
        gt_boxes = torch.from_numpy(blobs['gt_boxes']).to(device).unsqueeze(0)

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
        average_gradients(net)
        # torch.nn.utils.clip_grad_norm_(net.parameters(),0.5)
        optimizer.step()
        loss_sum += loss.item()

        timer["forward"].toc()

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

            print("\n------------------------Rank {} iteration {}-----------{} left---------\n "
                  "Average per iter: {:.4f} second.   ETA: {:.4f} hours\n"
                  "Average data load time: {:.4f}\n"
                  "loss:{}\nm3 cls:{}\nm3 box:{}\nm2 cls:{}\nm2 box:{}"
                  "\nm1 cls:{}\nm1 box:{}"
                  .format(rank,
                          iter,
                          max_iters - iter,
                          timer["forward"].average_time,
                          (max_iters - iter) * (timer["forward"].average_time) / (60 * 60),
                          timer["data"].average_time,
                          loss_average,
                          m3_ssh_cls_loss_average,
                          m3_bbox_loss_average,
                          m2_ssh_cls_loss_average,
                          m2_bbox_loss_average,
                          m1_ssh_cls_loss_average,
                          m1_bbox_loss_average
                          ))
            timer["forward"].reset()
            timer["data"].reset()

            # save_check_point(saved_model_path, iter, loss, net, optimizer)

        if iter % cfg.TRAIN.CHECKPOINT == 0 and int(rank) == 0:
            save_check_point(arg.model_save_path, iter, loss, net, optimizer)
            print("check point saved")


def init_processes(rank, size, gpu_id, arg, cfg, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = arg.master_ip
    os.environ['MASTER_PORT'] = arg.master_port
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    # init_method = 'tcp://'+arg.master_ip+':'+arg.master_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, gpu_id, arg, cfg)


if __name__ == '__main__':
    arg = parser()
    gpu_list = arg.gpu_ids.split(',')
    gpus = [int(i) for i in gpu_list]

    # size = len(gpus)
    # processes = []
    # for rank in range(size):
    #     t = Process(target=init_processes, args=(arg.rank, size, gpus[rank], arg, cfg))
    #     processes.append(t)
    #     t.start()
    #
    # for one_process in processes:
    #     one_process.join()
    size = arg.world_size
    init_processes(arg.rank, size, gpus[0], arg, cfg)
