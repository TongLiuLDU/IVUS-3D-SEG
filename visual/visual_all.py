#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化脚本：比较不同模型的分割结果
创建n×6的矩阵图片，6列分别对应：gt、segformer、swin_unetr、unetr、unetrpp、StripFormer_config9
"""

import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def get_available_slices(visual_dir):
    """获取所有可用的切片名称"""
    gt_dir = os.path.join(visual_dir, 'gt')
    all_slices = set()
    
    # 遍历所有病例文件夹
    for case_dir in os.listdir(gt_dir):
        case_path = os.path.join(gt_dir, case_dir)
        if os.path.isdir(case_path):
            # 获取该病例下的所有切片
            png_files = glob.glob(os.path.join(case_path, "*.png"))
            for png_file in png_files:
                slice_name = os.path.basename(png_file)
                all_slices.add(slice_name)
    
    return sorted(list(all_slices))

def find_image_path(visual_dir, method, slice_name):
    """根据方法名和切片名查找对应的图片路径"""
    method_dir = os.path.join(visual_dir, method)
    if not os.path.exists(method_dir):
        return None
    
    # 遍历所有病例文件夹寻找匹配的切片
    for case_dir in os.listdir(method_dir):
        case_path = os.path.join(method_dir, case_dir)
        if os.path.isdir(case_path):
            image_path = os.path.join(case_path, slice_name)
            if os.path.exists(image_path):
                return image_path
    
    return None

def visualize_comparison(visual_dir, slice_names, output_path="comparison_visualization.png", 
                        figsize_per_image=(3, 3), wspace=0.1, hspace=0.1):
    """
    创建比较可视化图片
    
    Args:
        visual_dir: visual目录路径
        slice_names: 要可视化的切片名称列表
        output_path: 输出图片路径
        figsize_per_image: 每个子图的大小
        wspace: 子图之间的宽度间距（0-1之间，0.1表示子图宽度的10%）
        hspace: 子图之间的高度间距（0-1之间，0.1表示子图高度的10%）
    """
    
    # 定义方法列表和对应的显示名称
    methods = ['gt','nnUNetTrainer_segformer3d', 'nnUNetTrainer_unetr','nnUNetTrainer_swinunetr', 'nnUNetTrainer_swinunetreffidec', 'nnUNetTrainer_unetrpp','my_model']
    method_display_names = ['Ground Truth', 'SegFormer3D', 'UNETR', 'SwinUNETR', 'SwinUNETRv2_EffiDec3D', 'UNETR++', 'ours']
    
    n_slices = len(slice_names)
    n_methods = len(methods)
    
    # 创建图形
    fig, axes = plt.subplots(n_slices, n_methods, 
                            figsize=(n_methods * figsize_per_image[0], n_slices * figsize_per_image[1]))
    
    # 确保axes总是2D数组
    if n_slices == 1 and n_methods == 1:
        axes = np.array([[axes]])
    elif n_slices == 1:
        axes = axes.reshape(1, -1)
    elif n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    # 遍历每个切片和方法
    for i, slice_name in enumerate(slice_names):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            
            # 查找图片路径
            image_path = find_image_path(visual_dir, method, slice_name)
            
            if image_path and os.path.exists(image_path):
                # 加载并显示图片
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
                
                # 为第一行添加列标题
                if i == 0:
                    ax.set_title(method_display_names[j], fontsize=12, fontweight='bold')
                    
                # 为第一列添加行标题（切片名）
                if j == 0:
                    # 提取切片编号作为行标签
                    slice_num = slice_name.replace('.png', '').split('_')[-1]
                    ax.set_ylabel(f'Slice {slice_num}', fontsize=10, fontweight='bold')
            else:
                # 如果图片不存在，显示空白并标注
                ax.text(0.5, 0.5, 'Image\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.set_facecolor('lightgray')
                ax.axis('off')
                
                if i == 0:
                    ax.set_title(method_display_names[j], fontsize=12, fontweight='bold')
                if j == 0:
                    slice_num = slice_name.replace('.png', '').split('_')[-1]
                    ax.set_ylabel(f'Slice {slice_num}', fontsize=10, fontweight='bold')
    
    # 调整布局和间距
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化结果已保存到: {output_path}")

def quick_visualize(slice_names, output_name=None, visual_dir=None, 
                   figsize_per_image=(3, 3), wspace=0.1, hspace=0.1):
    """
    快速可视化函数
    
    Args:
        slice_names: 切片名称列表或单个切片名称
        output_name: 输出文件名（可选）
        visual_dir: visual目录路径（可选）
        figsize_per_image: 每个子图的大小
        wspace: 子图之间的宽度间距（0-1之间）
        hspace: 子图之间的高度间距（0-1之间）
    """
    if visual_dir is None:
        visual_dir = '/home/files/liutong/xcodd/nnunetData/nnUNet_results/visual'
    
    # 处理输入
    if isinstance(slice_names, str):
        slice_names = [slice_names]
    
    # 设置输出文件名
    if output_name is None:
        if len(slice_names) == 1:
            slice_num = slice_names[0].replace('.png', '').split('_')[-1]
            output_name = f'comparison_slice_{slice_num}.png'
        else:
            output_name = f'comparison_{len(slice_names)}_slices.png'
    
    # 执行可视化
    print(f"正在可视化 {len(slice_names)} 个切片...")
    visualize_comparison(visual_dir, slice_names, output_name, figsize_per_image, wspace, hspace)

def visualize_gt_samples(visual_dir, num_samples=5, output_path="gt_samples.png", 
                         figsize_per_image=(3, 3), seed=None, wspace=0.1, hspace=0.1,slice_path=None):
    """
    随机可视化几个GT图片
    """
    # 获取所有可用切片
    all_slices = get_available_slices(visual_dir)
    
    if not all_slices:
        print("未找到切片")
        return

    # 随机选择
    if seed is not None:
        random.seed(seed)
    if slice_path is not None:
        selected_slices = slice_path
    else:
        if len(all_slices) > num_samples:
            selected_slices = random.sample(all_slices, num_samples)
        else:
            selected_slices = all_slices
    
    print(f"随机选择的GT切片: {selected_slices}")

    # 创建图形
    fig, axes = plt.subplots(1, len(selected_slices), 
                            figsize=(len(selected_slices) * figsize_per_image[0], figsize_per_image[1]))
    
    # 确保axes是列表
    if len(selected_slices) == 1:
        axes = [axes]
    
    for i, slice_name in enumerate(selected_slices):
        ax = axes[i]
        image_path = find_image_path(visual_dir, 'gt', slice_name)
        
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.axis('off')
            # 简化标题，只显示切片编号
            # slice_num = slice_name.replace('.png', '').split('_')[-1]
            # ax.set_title(f"Slice {slice_num}", fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
            ax.axis('off')

    # 调整布局和间距
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=0, right=1, bottom=0, top=1)
    # plt.tight_layout() # tight_layout可能会覆盖subplots_adjust的效果，根据需要选择
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show() 
    print(f"GT可视化结果已保存到: {output_path}")
















if __name__ == "__main__":
    # 直接运行的配置
    visual_dir = '/home/files/liutong/a1/xcodd/IVUS/visual/result'
    
    print("=== 医学图像分割结果可视化 ===")
    print()
    
    # 检查visual目录是否存在
    if not os.path.exists(visual_dir):
        print(f"错误：visual目录不存在: {visual_dir}")
        exit(1)
    
    # 获取可用的切片
    print("正在扫描可用的切片...")
    available_slices = get_available_slices(visual_dir)
    print(f"找到 {len(available_slices)} 个切片")
    print()
    
    if len(available_slices) == 0:
        print("没有找到可用的切片文件！")
        exit(1)
    

    
    # 你可以在这里修改要可视化的切片
    # 示例1: 可视化单个切片
    slice_to_visualize = "185118-Exam2_006_slice_032.png"  # 修改这里来选择要可视化的切片
    
    # 自定义间距参数（可以根据需要调整）
    wspace = 0.01  # 宽度间距：0.02表示子图宽度的2%（较小间距）
    hspace = 0.01  # 高度间距：0.02表示子图高度的2%（较小间距）
    figsize = (2.5, 2.5)  # 每个子图的大小

    
    
    slices = ["185118-Exam2_006_slice_105.png",available_slices[567],"296674-Exam2_008_slice_022.png",available_slices[2111],]
    print("可视化前3个切片的对比:")
    # quick_visualize(slices, "multi_slice_comparison.png", 
    #                 figsize_per_image=figsize, wspace=wspace, hspace=hspace,visual_dir=visual_dir)

    # 新增调用
    print("\n=== 随机可视化GT图片 ===")
    # slice_path = 
    slice_path = ['6211912-Exam1_071_slice_906.png', '375674-Exam5_070_slice_252.png', '375674-Exam3_068_slice_112.png']
    visualize_gt_samples(visual_dir, num_samples=3, output_path="random_gt_samples.png", seed=42, wspace=0.005, hspace=0.01,slice_path=slice_path)

