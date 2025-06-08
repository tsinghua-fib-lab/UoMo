import argparse

import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models_with_mask_scale import UoMo_models
from train import TrainLoop
import setproctitle
import torch
from DataLoader import data_load_main
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusion import create_diffusion
import datetime



def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")


def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        task = 'short64',
        dataset='TrafficNC*TrafficSD*TrafficNJ',  # ,#*TrafficSD*TrafficNJ
        mask_strategy=['random_masking', 'generation_masking', 'short_long_temporal_masking'],  # 'random'
        length0 = 64,
        batch_size=256,
        early_stop = 20,
        weight_decay=1e-4,
        log_interval=20,
        total_epoches = 100,
        mask_ratio = 0.5,
        lr_anneal_steps = 500,
        patch_size = 1,
        t_patch_size = 1,
        clip_grad = 1,
        min_lr = 1e-5,
        stage = 0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')


def print_memory_usage(layer, input, output):
    print(f'{layer}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated')



def main():

    th.autograd.set_detect_anomaly(True)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.dataset, args.total_epoches))
    setup_init(100)

    data, test_data, val_data, args.scaler = data_load_main(args)

    args.folder = 'Len{}_{}_Pretrain/'.format(args.length0, args.dataset.replace('*', '_'))
    args.model_path = f'./experiments/{args.folder}'
    logdir = "./logs/{}".format(args.folder)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    print('start data load')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    model = UoMo_models['UoMo-S'](
        args=args,
        in_channels = args.t_patch_size * args.patch_size * args.patch_size,
        depth=8,
        hidden_size=512,
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total par (M)', total_params/ 1000000.0)

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        diffusion = diffusion,
        data=data,
        test_data=test_data, 
        val_data=val_data,
        device=device
    ).run_loop()


if __name__ == "__main__":
    main()