# test_dataset.py
import torch
from datasets.mot_rgbt import GTOTDataset
import datasets.transforms_rgbt as T
from PIL import ImageDraw

# 路径指向你刚才上传的 GTOT 文件夹
data_root = "data/GTOT" 

# 实例化
# 注意：这里我们手动模拟 args
dataset = GTOTDataset(data_root, transforms=T.make_rgbt_transforms('train'))

print(f"Dataset length: {len(dataset)}")

# 取出一个样本
img_rgb, img_t, target = dataset[0]

print("RGB Shape:", img_rgb.shape)
print("Thermal Shape:", img_t.shape) # 应该是 [1, H, W] 或 [3, H, W] 取决于 transforms
print("Target:", target)

# 反归一化并保存图片检查对齐情况
# ... (这一步是可选的，用来肉眼看 bbox 是否歪了)