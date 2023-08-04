import os
import numpy as np
import argparse
import datetime
import time
import json
from pathlib import Path
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from dataloaders import get_dataloader
from models import get_model
from schedulers import get_scheduler
from losses import get_loss
from optimizers import get_optimizer
from engine import *
from torch.optim.swa_utils import AveragedModel, SWALR
from module.weight_methods import WeightMethods
from collections import defaultdict


# 명령어:
# python train.py --dataset 'ldctiqa_original' --batch-size 40 --num_workers 24 --model 'EfficientNetB7' --loss 'l2_ridge_loss' --optimizer 'adamw' --scheduler 'poly_lr' --epochs 1000 --lr 1e-4 --multi-gpu-mode 'DataParallel' --gpu-ids '4, 5' --checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/checkpoints/230604_EfficientNetB7' --save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/predictions/230604_EfficientNetB7' --from-pretrained '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/checkpoints/230604_EfficientNetB7/epoch_182_checkpoint.pth' --resume '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/checkpoints/230604_EfficientNetB7/epoch_182_checkpoint.pth'

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deep-Learning Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',  default='ldctiqa',  type=str, help='dataset name')    
    parser.add_argument('--batch-size',  default=72, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # Model parameters
    parser.add_argument('--model',  default='Unet',  type=str, help='model name')    
    parser.add_argument('--method', default='', help='multi-task weighting name')

    # Loss parameters
    parser.add_argument('--loss',  default='dice_loss',  type=str, help='loss name')

    # SWA parameters
    parser.add_argument('--swa', default="FALSE", type=utils.str2bool, help='swa learning')    
    parser.add_argument('--swa-lr', type=float, default=0.05, metavar='LR', help='swa learning rate')
    parser.add_argument('--swa-start-epoch', type=int, default=10, metavar='N', help='swa start epoch')    

    # Training parameters - Optimizer, LR, Scheduler, Epoch
    parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    parser.add_argument('--scheduler', default='poly_lr', type=str, metavar='scheduler', help='scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs', default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR, if scheduler supports')    

    # Continue Training (Resume)
    parser.add_argument('--from-pretrained',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')

    # Save setting
    parser.add_argument('--checkpoint-dir', default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir', default='', help='path where to prediction PNG save')
    parser.add_argument('--memo', default='', help='memo for script')
    return parser

# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
np.random.seed(random_seed)
random.seed(random_seed)

def main(args):
    start_epoch = 0
    utils.print_args(args)
    device = torch.device(args.device)

    # Dataset
    train_loader = get_dataloader(name=args.dataset, mode='train', batch_size=args.batch_size, num_workers=args.num_workers)   
    valid_loader = get_dataloader(name=args.dataset, mode='valid', batch_size=1,               num_workers=int(args.num_workers//4))

    # Model
    model = get_model(name=args.model)

    # Pretrained
    if args.from_pretrained:
        print("Loading... Pretrained")
        checkpoint = torch.load(args.from_pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Multi-GPU & CUDA
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)         
        model = model.to(device)

    else :
        model = model.to(device)

    # Optimizer & LR Schedule & Loss
    if (args.method) and (not args.resume):
        # weight method
        weight_methods_parameters = defaultdict(dict)
        weight_methods_parameters.update(dict(nashmtl=dict(update_weights_every=1, optim_niter=20), stl=dict(main_task=0), cagrad=dict(c=0.4), dwa=dict(temp=2.0))) # ref: https://github.com/AvivNavon/nash-mtl/tree/7cc1694a276ca6f2f9426ab18b8698c786bff4f0
        weight_method = WeightMethods(args.method, n_tasks=4, device=device, **weight_methods_parameters[args.method])
        optimizer = torch.optim.AdamW([
            dict(params=model.parameters(),         lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False),
            dict(params=weight_method.parameters(), lr=0.025,   betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)])
    else:
        optimizer = get_optimizer(name=args.optimizer, model=model, lr=args.lr)

    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, warm_up_epoch=10, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=1e-6)
    criterion = get_loss(name=args.loss)


    # Resume
    if args.resume:
        print("Loading... Resume")
        start_epoch, model, optimizer, scheduler = utils.load_checkpoint(model, optimizer, scheduler, filename=args.resume)

    # Tensorboard
    tensorboard = SummaryWriter(args.checkpoint_dir + '/runs')

    # Etc traing setting
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # SWA
    if args.swa:
        print("SWA Training...")
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    # Whole Loop Train & Valid 
    for epoch in range(start_epoch, args.epochs):
        
        if args.model == 'Cell_UNet' or args.model == 'Cell_MaxViT_UNet_DET':
            train_stats = train_cell_loop_fn(train_loader, model, criterion, optimizer, device, epoch)
            print('==> Averaged [Train] stats: ' + str(train_stats))
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_cell_loop_fn(valid_loader, model, criterion, device, epoch, args.save_dir)
            print('==> Averaged [Valid] stats: ' + str(valid_stats))
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)

        elif args.model == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC':
            train_stats = train_cell_mtl_det_seg_rec_loop_fn(train_loader, model, criterion, optimizer, device, epoch)
            print('==> Averaged [Train] stats: ' + str(train_stats))
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_cell_mtl_det_seg_rec_loop_fn(valid_loader, model, criterion, device, epoch, args.save_dir)
            print('==> Averaged [Valid] stats: ' + str(valid_stats))
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)

        elif args.model == 'Cell_MaxViT_UNet_MTL_DET_SEG_POI':
            train_stats = train_cell_mtl_det_seg_poi_loop_fn(train_loader, model, criterion, optimizer, device, epoch)
            print('==> Averaged [Train] stats: ' + str(train_stats))
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_cell_mtl_det_seg_poi_loop_fn(valid_loader, model, criterion, device, epoch, args.save_dir)
            print('==> Averaged [Valid] stats: ' + str(valid_stats))
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)                

        elif args.model == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI':
            train_stats = train_cell_mtl_det_seg_rec_poi_loop_fn(train_loader, model, criterion, optimizer, device, epoch)
            print('==> Averaged [Train] stats: ' + str(train_stats))
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_cell_mtl_det_seg_rec_poi_loop_fn(valid_loader, model, criterion, device, epoch, args.save_dir)
            print('==> Averaged [Valid] stats: ' + str(valid_stats))
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)

        elif args.model == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method':
            train_stats = train_cell_mtl_det_seg_rec_poi_method_loop_fn(train_loader, model, criterion, optimizer, device, epoch, weight_method)
            print('==> Averaged [Train] stats: ' + str(train_stats))
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_cell_mtl_det_seg_rec_poi_method_loop_fn(valid_loader, model, criterion, device, epoch, args.save_dir)
            print('==> Averaged [Valid] stats: ' + str(valid_stats))
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)


        # LR scheduler Update
        if args.swa:            
            if epoch > args.swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()        
            else:
                scheduler.step()

        else:
            scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, args.checkpoint_dir + '/epoch_' + str(epoch) + '_checkpoint.pth')

        # Log text
        log_stats = {**{f'{k}': v for k, v in train_stats.items()}, 
                    **{f'{k}': v for k, v in valid_stats.items()}, 
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']}

        with open(args.checkpoint_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Finish
    if args.swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model) # Update bn statistics for the swa_model at the end
        torch.save({'epoch': epoch, 'model_state_dict': swa_model.state_dict()}, args.checkpoint_dir + '/SWA_checkpoint.pth') # Save checkpoint

    tensorboard.close()
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
        
    # Make folder if not exist
    os.makedirs(args.checkpoint_dir, exist_ok =True)
    os.makedirs(args.checkpoint_dir + "/args", exist_ok =True)
    os.makedirs(args.save_dir, exist_ok =True)

    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
       
    main(args)
