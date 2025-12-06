import os
from pathlib import Path

# 定义项目根目录名称
project_name = "thermalTracker"

# 定义目录结构
structure = {
    "configs": ["motr_rgbt.sh"],      # 存放实验配置和启动脚本
    "datasets": [                     # 数据加载相关
        "__init__.py",
        "mot_rgbt.py",                # [核心] RGB+Thermal 数据集加载类
        "transforms_rgbt.py",         # [核心] 双模态数据增强
        "coco_eval.py"                # 评测工具（通常复用）
    ],
    "models": [                       # 模型定义
        "__init__.py",
        "backbone.py",                # 双流骨干网络
        "motr.py",                    # 模型入口
        "transformer.py",             # [核心] Cross-Modality Attention 实现
        "matcher.py",                 # 匈牙利匹配
        "position_encoding.py",       # 位置编码                     # [重要] 存放 CUDA 算子 (C++源码)
    ],
    "util": [                         # 工具类
        "__init__.py",
        "misc.py",                    # 分布式训练、日志等杂项
        "box_ops.py",                 # 边框计算 IoU 等
        "plot_utils.py"               # 可视化工具
    ],
    ".": [                            # 根目录文件
        "main.py",                    # 训练主入口
        "engine.py",                  # 训练与验证的 epoch 循环逻辑
        "README.md",
        ".gitignore",
        "requirements.txt"
    ]
}

def create_structure():
    root = Path(os.getcwd())
    
    print(f"🚀 Initializing {project_name} structure...")

    for folder, files in structure.items():
        # 处理根目录和子目录
        if folder == ".":
            current_dir = root
        else:
            current_dir = root / folder
            current_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {current_dir}")

        # 创建文件
        for file_name in files:
            file_path = current_dir / file_name
            if not file_path.exists():
                # 如果是 ops 目录，通常需要放 C++ 代码，这里先跳过文件的创建，手动复制
                if folder == "models" and file_name == "ops":
                    continue
                    
                with open(file_path, "w", encoding="utf-8") as f:
                    # 为不同的文件写入一些基础注释
                    if file_name.endswith(".py"):
                        f.write(f"# {file_name} - Part of {project_name}\n\n")
                    if file_name == "README.md":
                        f.write(f"# {project_name}\n\nMultimodal (RGB+T) Multi-Object Tracking based on MOTR.\n")
                    if file_name == ".gitignore":
                        pass # 稍后单独写入
                print(f"  -> Created file: {file_name}")

    print("\n✅ Project structure setup complete!")

if __name__ == "__main__":
    create_structure()