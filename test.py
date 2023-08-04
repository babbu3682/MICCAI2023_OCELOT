import os
import numpy as np
import argparse
import datetime
import time
import json
from pathlib import Path
import random
import torch
import utils
from dataloaders import get_dataloader
from models import get_model
from schedulers import get_scheduler
from losses import get_loss
from optimizers import get_optimizer
from engine import test_loop_fn
from torch.optim.swa_utils import AveragedModel

# 명령어:
# python test.py --dataset 'ldctiqa_original' --batch-size 1 --num_workers 8 --model 'EfficientNetB7' --loss 'l2_loss' --gpu-ids '3' --checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/checkpoints/230604_EfficientNetB7' --save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/predictions/230604_EfficientNetB7' --resume '/workspace/sunggu/0.Challenge/MICCAI2023_LDCTIQA/checkpoints/230604_EfficientNetB7/epoch_47_checkpoint.pth'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Deep-Learning Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',  default='ldctiqa',  type=str, help='dataset name')    
    parser.add_argument('--batch-size',  default=1, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # Model parameters
    parser.add_argument('--model',  default='Unet',  type=str, help='model name')    

    # Loss parameters
    parser.add_argument('--loss',  default='dice_loss',  type=str, help='loss name')        
    
    # SWA parameters
    parser.add_argument('--swa', default="FALSE", type=utils.str2bool, help='swa inference')
    parser.add_argument('--tta', default="FALSE", type=utils.str2bool, help='tta inference')

    # Continue Training (Resume)
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # '' = None

    # DataParrel or Single GPU train
    parser.add_argument('--gpu-ids', default='0', type=str, help='cuda_visible_devices')
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')

    # Save setting
    parser.add_argument('--checkpoint-dir', default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',   default='', help='path where to prediction PNG save')

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
    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataset
    test_loader = get_dataloader(name=args.dataset, mode='test', batch_size=1, num_workers=args.num_workers)

    # Model
    model = get_model(name=args.model)
    
    # CUDA
    model.to(device)

    # Loss
    criterion = get_loss(name=args.loss)

    # SWA
    if args.swa:
        swa_model = AveragedModel(model)

    # Resume
    if args.resume:
        if args.swa:
            checkpoint = torch.load(args.resume)            
            swa_model.load_state_dict(checkpoint['model_state_dict'])

        else:
            checkpoint = torch.load(args.resume)
            model_state_dict = utils.check_checkpoint_if_wrapper(checkpoint['model_state_dict'])
            model.load_state_dict(model_state_dict)

    # Etc traing setting
    start_time = time.time()

    test_stats = test_loop_fn(test_loader, model, criterion, device, args.save_dir, args.checkpoint_dir, args.tta)
    print('==> Averaged [TEST] stats: ' + str(test_stats))
        

    print('***********************************************')
    print("Finish...!")
    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()        

    os.makedirs(args.checkpoint_dir + "/args", exist_ok =True)

    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # CUDA setting
    os.environ["CUDA_DEVICE_ORDER"]    = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids   

    main(args)


