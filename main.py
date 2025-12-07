# main.py
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from models.motr import build as build_model
from engine import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('ThermalTracker Training Script', add_help=False)
    
    # 核心训练参数
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int) # 显存不够可改为 1
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--aux_loss', default=True, type=bool)

    # Loss coefficients
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/GTOT', type=str) # 确保路径正确
    parser.add_argument('--output_dir', default='output/rgbt_exp1', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    return parser

def main(args):
    print("🚀 Starting ThermalTracker Training...")
    print(args)
    
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Build Model
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Number of parameters: {n_parameters}")

    # 2. Build Optimizer (Backbone 和 Transformer 使用不同的学习率)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 3. Build Dataset
    dataset_train = datasets.build_dataset(image_set='train', args=args)
    # dataset_val = datasets.build_dataset(image_set='val', args=args) # 暂时用训练集测试

    print(f"📂 Training dataset size: {len(dataset_train)}")
    
    # 使用自定义的 collate_fn
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                                   shuffle=True, collate_fn=utils.collate_fn, 
                                   num_workers=args.num_workers)

    # 4. Training Loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            max_norm=args.clip_max_norm)
        
        lr_scheduler.step()
        
        # Save checkpoint
        if args.output_dir:
            checkpoint_path = output_dir / f'checkpoint{epoch:04}.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)