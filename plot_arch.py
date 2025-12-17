import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='#E1F5FE', edge='#01579B', fontsize=10):
    # 绘制带圆角的框
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                 linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#000000')
    return x+w, y+h/2 # 返回右侧连接点

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 1. Inputs
    draw_box(ax, 0.5, 5.5, 2, 1, "RGB Image\n(3xHxW)", color='#FFF3E0', edge='#E65100')
    draw_box(ax, 0.5, 1.5, 2, 1, "Thermal Image\n(1xHxW)", color='#FCE4EC', edge='#880E4F')

    # 2. Preprocessing (Channel Adapter)
    # RGB直接过
    draw_arrow(ax, 2.55, 6.0, 3.5, 6.0)
    # Thermal 经过适配
    draw_arrow(ax, 2.55, 2.0, 3.5, 2.0)
    draw_box(ax, 3.5, 1.75, 1.5, 0.5, "Channel\nAdapter\n(Repeat x3)", color='#E0E0E0', edge='#424242', fontsize=8)

    # 3. Siamese Backbone (Shared Weights)
    # 画一个大虚线框表示 Siamese
    siamese_bg = patches.Rectangle((5.5, 1.0), 2.5, 6.0, linewidth=1, linestyle='--', edgecolor='gray', facecolor='none')
    ax.add_patch(siamese_bg)
    ax.text(6.75, 7.2, "Siamese Backbone\n(Shared Weights & Frozen BN)", ha='center', fontsize=9, style='italic')

    # 上路 ResNet
    draw_arrow(ax, 2.55, 6.0, 5.8, 6.0) # RGB line connect
    draw_box(ax, 5.8, 5.5, 2, 1, "ResNet-50\n(RGB)", color='#E8F5E9', edge='#1B5E20')
    
    # 下路 ResNet
    draw_arrow(ax, 5.0, 2.0, 5.8, 2.0)
    draw_box(ax, 5.8, 1.5, 2, 1, "ResNet-50\n(Thermal)", color='#E8F5E9', edge='#1B5E20')

    # 4. Feature Fusion / Input to Transformer
    draw_arrow(ax, 7.85, 6.0, 9.0, 4.5)
    draw_arrow(ax, 7.85, 2.0, 9.0, 3.5)

    # 5. Transformer
    draw_box(ax, 9.0, 3.0, 2, 2, "Transformer\nEncoder-Decoder\n(MOTR)", color='#F3E5F5', edge='#4A148C')

    # 6. Heads
    draw_arrow(ax, 11.05, 4.0, 12.0, 4.0)
    draw_box(ax, 12.0, 3.5, 1.5, 1, "Prediction\nHeads\n(Box & Class)", color='#FFF8E1', edge='#FF6F00')

    # 7. Output
    draw_arrow(ax, 13.55, 4.0, 13.9, 4.0)
    
    # Title or Final Box
    # ax.text(14.0, 4.0, "Tracking\nResult", ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Figure1_Architecture.png", dpi=300, bbox_inches='tight')
    print("✅ 图1已生成：Figure1_Architecture.png")

if __name__ == "__main__":
    main()
    