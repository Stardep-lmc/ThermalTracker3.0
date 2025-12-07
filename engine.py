# engine.py
import math
import sys
import torch
import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # 遍历 DataLoader
    for samples_rgb, samples_thermal, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 1. 数据移至 GPU
        samples_rgb = samples_rgb.to(device)
        samples_thermal = samples_thermal.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. 前向传播 (Forward)
        outputs = model(samples_rgb, samples_thermal)
        
        # 3. 计算 Loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # 加权求和
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # 4. 反向传播 (Backward)
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪 (防止爆炸)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        optimizer.step()

        # 5. 日志记录
        metric_logger.update(loss=loss_value, **loss_dict)
        # [修改] 加上 .0 变成浮点数
        metric_logger.update(class_error=loss_dict.get('class_error', 0.0)) 
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples_rgb, samples_thermal, targets in metric_logger.log_every(data_loader, 10, header):
        samples_rgb = samples_rgb.to(device)
        samples_thermal = samples_thermal.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples_rgb, samples_thermal)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict # 单卡省略 reduce
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
                             
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}