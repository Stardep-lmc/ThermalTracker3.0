# datasets/__init__.py
import torch.utils.data
from .mot_rgbt import build as build_mot_rgbt

def build_dataset(image_set, args):
    # 这里是一个分发入口，目前只支持 GTOT (mot_rgbt)
    # 如果 args.dataset_file == 'mot_rgbt' ... (如果有多个数据集可以在这写 if)
    return build_mot_rgbt(image_set, args)