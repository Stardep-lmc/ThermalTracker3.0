import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    # 这里填你刚才跑出来的最好的结果文件夹
    # 比如 output/vis_result_final_brute/Tricycle 或者 output/vis_result_twostream/Tricycle
    # 请根据你实际生成的文件夹修改下面这行！
    source_dir = "output/vis_result_final/Tricycle" 
    
    # 如果找不到，尝试找 brute force 的目录
    if not os.path.exists(source_dir):
        source_dir = "output/vis_result_final_brute/Tricycle"
        if not os.path.exists(source_dir):
             print(f"❌ 找不到文件夹: {source_dir}，请修改脚本里的 source_dir 为你 demo.py 的输出路径")
             return

    # 我们要展示的帧索引
    target_frames = [0, 50, 100]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    print(f"正在从 {source_dir} 读取图片...")

    for i, frame_idx in enumerate(target_frames):
        # 构造文件名，例如 0000.jpg, 0050.jpg
        fname = f"{frame_idx:04d}.jpg"
        path = os.path.join(source_dir, fname)
        
        if os.path.exists(path):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {frame_idx}", fontsize=14, fontweight='bold')
            axes[i].axis('off')
            
            # 加个边框让图更好看
            for spine in axes[i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
        else:
            print(f"⚠️ 警告: 找不到图片 {path}")
            axes[i].text(0.5, 0.5, "Image Not Found", ha='center', va='center')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("Figure2_TrackingResults.png", dpi=300, bbox_inches='tight')
    print("✅ 图2已生成：Figure2_TrackingResults.png")

if __name__ == "__main__":
    main()