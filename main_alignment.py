import argparse

import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models_with_mask_scale_align import UoMo_models
from train_align import TrainLoop
import setproctitle
import torch
from DataLoader_align import data_load_main
from utils import *
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusion_align import create_diffusion





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
        lr=1e-4,
        task='short64',
        length0=64,
        early_stop=20,
        weight_decay=1e-4,
        batch_size=128,
        log_interval=20,
        total_epoches=100,
        lr_anneal_steps=500,
        patch_size=1,
        t_patch_size=1,
        clip_grad=1,
        mask_strategy=['random_masking','generation_masking', 'short_long_temporal_masking'],  # 'random'
        min_lr=1e-5,
        dataset='TrafficNC*TrafficSD*TrafficNJ',  # ,#*TrafficSD*TrafficNJ
        stage=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)

    data, test_data, val_data, args.scaler = data_load_main(args)

    args.folder = 'Len{}_data_{}_Finetuning/'.format(args.length0, args.dataset.replace('*', '_'))
    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    print('start data load')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev()


    model = UoMo_models['UoMo-S'](
        args=args,
        in_channels = args.t_patch_size * args.patch_size * args.patch_size,
        depth=8,
        hidden_size=512,
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="")


    args.finetuing_path = 'Len64_TrafficNJ_Pretrain'#replace with your pretraining path!
    model.load_state_dict(torch.load(f'./experiments/{args.finetuing_path}/model_save/model_best.pkl',map_location=device), strict=False)
    model = model.to(device)

    for k, param in model.named_parameters():
        if 'attn' in k: #and 'mlp' in k:
            param.requires_grad = False


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
