#!/bin/bash
 (python train_dist.py --gpu='0' --rank=0 --world_size=2 &
 python train_dist.py --gpu='1' --rank=1 --world_size=2)
