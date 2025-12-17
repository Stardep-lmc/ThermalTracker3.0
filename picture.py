import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 顶会级配色方案 (保持不变) ---
C_RGB_FACE = '#E3F2FD'      
C_RGB_EDGE = '#1565C0'      
C_TH_FACE  = '#FBE9E7'      
C_TH_EDGE  = '#D84315'      
C_ADAPTER  = '#D1C4E9'      
C_TRANS    = '#E8EAF6'      
C_HEAD     = '#FFF3E0'      
C_TEXT     = '#37474F'      
C_SHARED   = '#ECEFF1'      

# --- 全局字体设置 (关键修改) ---
FONT_S_SMALL = 10  # 小字 (注释)
FONT_S_NORM  = 12  # 正文 (模块名)
FONT_S_BOLD  = 14  # 强调 (大标题)
LINE_WIDTH   = 1.5 # 线条加粗

def draw_stack(ax, x, y, w, h, num_slices, color_face, color_edge, label=None, fontsize=FONT_S_NORM):
    """绘制层叠切片"""
    depth_x, depth_y = 0.05, 0.05
    total_dx = num_slices * depth_x
    total_dy = num_slices * depth_y
    
    for i in range(num_slices, -1, -1):
        cx = x + i * depth_x
        cy = y + i * depth_y
        lw = LINE_WIDTH if i == 0 else 0.5
        alpha = 1.0 if i == 0 else 0.9
        
        rect = patches.Rectangle((cx, cy), w, h, linewidth=lw, edgecolor=color_edge, facecolor=color_face, alpha=alpha, zorder=100-i)
        ax.add_patch(rect)
    
    if label:
        # 文字位置稍微下移，字体加大
        ax.text(x + w/2 + total_dx/2, y - 0.4, label, ha='center', va='top', fontsize=fontsize, color=C_TEXT, fontweight='bold')
    
    return (x+w, y+h/2), (x+w+total_dx, y+h/2+total_dy)

def draw_operation_circle(ax, x, y, symbol, color='#FFF9C4', size=0.4):
    circle = patches.Circle((x, y), size, facecolor=color, edgecolor='black', linewidth=LINE_WIDTH, zorder=200)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=FONT_S_BOLD, fontweight='bold', zorder=201)
    return x, y

def draw_arrow(ax, p1, p2, curve=0.0, label=None, style='simple'):
    x1, y1 = p1
    x2, y2 = p2
    if x2 > x1: x2 -= 0.1
    
    # 箭头加粗
    arrow_style = "Simple,tail_width=0.8,head_width=5,head_length=6"
    connection = f"arc3,rad={curve}"
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=connection, 
                                    arrowstyle=arrow_style, color='#546E7A', lw=0, zorder=300) # lw=0 因为 Simple 样式自带宽度
    ax.add_patch(arrow)
    
    if label:
        mid_x = (x1+x2)/2
        mid_y = (y1+y2)/2 + (0.3 if curve==0 else curve)
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=FONT_S_SMALL, style='italic', color='#455A64', backgroundcolor='white')

def main():
    # [关键优化] 缩小画布尺寸，这就相当于变相放大了字体
    # 宽度设为 16 (适合跨栏大图)，高度 7
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ==========================================
    # 1. Input Details
    # ==========================================
    # RGB
    for i in range(3):
        offset = i * 0.15
        draw_stack(ax, 0.5 + offset, 7.5 + offset, 1.2, 1.2, 0, C_RGB_FACE, C_RGB_EDGE)
    ax.text(1.1, 7.1, "RGB Input\n(3 $\\times$ H $\\times$ W)", ha='center', va='top', fontsize=FONT_S_NORM)
    rgb_out_pt = (1.8, 8.1)

    # Thermal
    draw_stack(ax, 0.8, 2.5, 1.2, 1.2, 0, C_TH_FACE, C_TH_EDGE, "Thermal\n(1 $\\times$ H $\\times$ W)", fontsize=FONT_S_NORM)
    th_out_pt = (2.0, 3.1)

    # ==========================================
    # 2. Residual Thermal Adapter (RTA)
    # ==========================================
    # 背景框
    rta_bg = patches.FancyBboxPatch((2.5, 1.0), 3.8, 4.5, boxstyle="round,pad=0.1", 
                                    fc='#F3E5F5', ec='#BA68C8', ls='--', lw=LINE_WIDTH, zorder=0)
    ax.add_patch(rta_bg)
    ax.text(4.4, 5.1, "Residual Thermal Adapter\n(RTA)", ha='center', fontsize=FONT_S_BOLD, fontweight='bold', color='#7B1FA2')

    # Path A: Residual Identity
    add_x, add_y = 5.8, 3.1
    draw_arrow(ax, th_out_pt, (add_x, add_y), curve=-0.6, label="Identity")

    # Path B: Conv Expansion
    conv1_f, conv1_b = draw_stack(ax, 3.0, 2.7, 0.5, 0.8, 8, C_ADAPTER, '#7B1FA2', "Conv1\n(16ch)", fontsize=FONT_S_SMALL)
    draw_arrow(ax, th_out_pt, conv1_f)
    
    # ReLU
    relu_x, relu_y = 4.4, 3.1
    draw_operation_circle(ax, relu_x, relu_y, "f", size=0.3)
    draw_arrow(ax, conv1_b, (relu_x-0.3, relu_y))
    
    # 16->3 Conv
    conv2_f, conv2_b = draw_stack(ax, 4.9, 2.7, 0.5, 0.8, 2, C_ADAPTER, '#7B1FA2', "Conv2\n(3ch)", fontsize=FONT_S_SMALL)
    draw_arrow(ax, (relu_x+0.3, relu_y), conv2_f)

    # Add
    draw_operation_circle(ax, add_x, add_y, "+", size=0.3)
    draw_arrow(ax, conv2_b, (add_x-0.3, add_y))
    
    rta_out_pt = (add_x+0.4, add_y)

    # ==========================================
    # 3. Siamese Backbone
    # ==========================================
    shared_bg = patches.FancyBboxPatch((6.8, 0.5), 6.5, 9.0, boxstyle="round,pad=0.2", 
                                       fc=C_SHARED, ec='#B0BEC5', ls='-', lw=LINE_WIDTH, zorder=0)
    ax.add_patch(shared_bg)
    ax.text(10.0, 9.6, "Siamese Backbone (ResNet-50)", ha='center', fontsize=FONT_S_BOLD, fontweight='bold')
    ax.text(10.0, 9.2, "Shared Weights $\\theta$ & Frozen BN", ha='center', fontsize=FONT_S_NORM, style='italic', color='#546E7A')

    def draw_resnet_stream(y_base, color_f, color_e, input_pt):
        # 稍微加大方块尺寸，让字能放进去
        s1_f, s1_b = draw_stack(ax, 7.3, y_base, 1.1, 1.1, 3, color_f, color_e, "Stage1\n(/4)", fontsize=FONT_S_SMALL)
        draw_arrow(ax, input_pt, s1_f)
        
        s2_f, s2_b = draw_stack(ax, 9.2, y_base+0.1, 0.9, 0.9, 6, color_f, color_e, "Stage2\n(/8)", fontsize=FONT_S_SMALL)
        draw_arrow(ax, s1_b, s2_f)
        
        s3_f, s3_b = draw_stack(ax, 10.9, y_base+0.2, 0.7, 0.7, 9, color_f, color_e, "Stage3\n(/16)", fontsize=FONT_S_SMALL)
        draw_arrow(ax, s2_b, s3_f)
        
        s4_f, s4_b = draw_stack(ax, 12.4, y_base+0.25, 0.6, 0.6, 12, color_f, color_e, "Stage4\n(/32)", fontsize=FONT_S_SMALL)
        draw_arrow(ax, s3_b, s4_f)
        
        return s4_b

    rgb_final = draw_resnet_stream(7.5, C_RGB_FACE, C_RGB_EDGE, rgb_out_pt)
    th_final = draw_resnet_stream(2.5, C_TH_FACE, C_TH_EDGE, rta_out_pt)

    # 共享连接
    ax.annotate("", xy=(10.0, 7.5), xytext=(10.0, 3.5), arrowprops=dict(arrowstyle="<->", lw=2, color='#78909C', ls='--'))
    draw_operation_circle(ax, 10.0, 5.5, "W", size=0.6, color='white') # W for Weights

    # ==========================================
    # 4. Transformer
    # ==========================================
    trans_bg = patches.Rectangle((14.2, 2.0), 5.5, 6.5, fc=C_TRANS, ec='#7986CB', lw=LINE_WIDTH, zorder=0)
    ax.add_patch(trans_bg)
    ax.text(17.0, 8.1, "Transformer Decoder", ha='center', fontsize=FONT_S_BOLD, fontweight='bold', color='#283593')

    # Query
    q_f, q_b = draw_stack(ax, 14.8, 6.8, 0.6, 0.9, 2, '#FFF9C4', '#FBC02D', "Object\nQueries", fontsize=FONT_S_SMALL)
    
    # Self Attn
    sa_x, sa_y = 16.5, 6.8
    draw_operation_circle(ax, sa_x+0.8, sa_y+0.4, "Self\nAttn", color='#C5CAE9', size=0.7)
    draw_arrow(ax, q_b, (sa_x+0.1, sa_y+0.4))

    # Cross Attn
    ca_x, ca_y = 16.5, 4.5
    draw_operation_circle(ax, ca_x+0.8, ca_y+0.4, "Cross\nAttn", color='#FFCCBC', size=0.7)
    
    draw_arrow(ax, (sa_x+0.8, sa_y-0.3), (ca_x+0.8, ca_y+1.1)) # Self -> Cross
    
    # Fusion Arrows
    draw_arrow(ax, rgb_final, (ca_x+0.1, ca_y+0.6), curve=-0.1, label="RGB")
    draw_arrow(ax, th_final, (ca_x+0.1, ca_y+0.2), curve=0.1, label="Thermal")

    # FFN
    ffn_x, ffn_y = 16.5, 2.5
    draw_operation_circle(ax, ffn_x+0.8, ffn_y+0.4, "FFN", color='#C5CAE9', size=0.6)
    draw_arrow(ax, (ca_x+0.8, ca_y-0.3), (ffn_x+0.8, ffn_y+1.0))

    out_embed = (18.5, 3.0) # FFN Output

    # ==========================================
    # 5. Prediction Heads
    # ==========================================
    head_start_x = 20.2
    
    # Class Head
    draw_arrow(ax, (ffn_x+1.4, ffn_y+0.4), (head_start_x, 5.5), curve=-0.1)
    draw_stack(ax, head_start_x, 5.2, 0.8, 0.8, 3, C_HEAD, '#FF6F00', "Linear", fontsize=FONT_S_SMALL)
    draw_arrow(ax, (head_start_x+1.0, 5.5), (22.5, 5.5))
    ax.text(22.8, 5.5, "Class Score", ha='left', va='center', fontsize=FONT_S_NORM, fontweight='bold')

    # Box Head
    draw_arrow(ax, (ffn_x+1.4, ffn_y+0.4), (head_start_x, 2.5), curve=0.1)
    draw_stack(ax, head_start_x, 2.2, 0.8, 0.8, 3, C_HEAD, '#FF6F00', "MLP", fontsize=FONT_S_SMALL)
    draw_arrow(ax, (head_start_x+1.0, 2.5), (22.5, 2.5))
    ax.text(22.8, 2.5, "Box Coord\n(x,y,w,h)", ha='left', va='center', fontsize=FONT_S_NORM, fontweight='bold')

    # 保存时去白边
    plt.tight_layout()
    plt.savefig("Figure1_Paper_Ready_LargeFont.png", dpi=300, bbox_inches='tight')
    print("✅ 论文专用大字版已生成: Figure1_Paper_Ready_LargeFont.png")

if __name__ == "__main__":
    main()