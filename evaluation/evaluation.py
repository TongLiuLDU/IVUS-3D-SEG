import os
import numpy as np
import nibabel as nib
import argparse
import json
from copy import deepcopy
from medpy.metric.binary import hd
from multiprocessing import Pool
from functools import partial


def load_nifti_image(file_path):
    """加载NIfTI图像"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata(dtype='float32')  # 使用float32减少内存
        return data
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None


def calculate_hausdorff_distance(mask_ref, mask_pred):
    """计算Hausdorff距离"""
    mask_ref = mask_ref.astype(bool)
    mask_pred = mask_pred.astype(bool)
    
    if np.sum(mask_ref) == 0 and np.sum(mask_pred) == 0:
        return 0.0
    if np.sum(mask_ref) == 0 or np.sum(mask_pred) == 0:
        return float('inf')
    
    return hd(mask_ref, mask_pred)


def calculate_percentage_area_difference(mask_ref, mask_pred, ignore_mask=None):
    """计算Percentage of Area Difference (PAD)"""
    mask_ref = mask_ref.astype(bool)
    mask_pred = mask_pred.astype(bool)
    
    if ignore_mask is not None:
        use_mask = ~ignore_mask
        area_g = np.sum(mask_ref & use_mask)
        area_p = np.sum(mask_pred & use_mask)
    else:
        area_g = np.sum(mask_ref)
        area_p = np.sum(mask_pred)
    
    if area_g == 0:
        return 0.0 if area_p == 0 else float('inf')
    
    return abs(area_p - area_g) / area_g * 100


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label) -> np.ndarray:
    """根据nnUNet的实现，将标签转换为掩码"""
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    """完全按照nnUNet的实现计算混淆矩阵"""
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics_nnunet_style(reference_file: str, prediction_file: str, 
                                 labels_or_regions, ignore_label: int = None) -> dict:
    """完全按照nnUNet的方式计算指标"""
    # 加载图像
    seg_ref = load_nifti_image(reference_file)
    seg_pred = load_nifti_image(prediction_file)
    
    if seg_ref is None or seg_pred is None:
        return None
    
    # 转换为整数
    seg_ref = np.round(seg_ref).astype(np.int16)  # 使用int16减少内存
    seg_pred = np.round(seg_pred).astype(np.int16)
    
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        
        # 完全按照nnUNet的计算方式
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        
        results['metrics'][r]['FP'] = int(fp)
        results['metrics'][r]['TP'] = int(tp)
        results['metrics'][r]['FN'] = int(fn)
        results['metrics'][r]['TN'] = int(tn)
        results['metrics'][r]['n_pred'] = int(fp + tp)
        results['metrics'][r]['n_ref'] = int(fn + tp)
        
        # 计算HD和PAD
        print(f"  计算类别 {r} 的HD和PAD...")
        
        # 为HD计算裁剪到边界框以加速
        union = mask_ref | mask_pred
        if np.sum(union) == 0:
            hd_value = 0.0
        else:
            coords = np.argwhere(union)  # 使用argwhere代替where以提高效率
            if len(coords) == 0:
                hd_value = 0.0
            else:
                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                padding = 10  # 增加填充以确保覆盖
                min_coords = np.maximum(0, min_coords - padding)
                max_coords = np.minimum(np.array(union.shape) - 1, max_coords + padding)
                slices = tuple(slice(minc, maxc + 1) for minc, maxc in zip(min_coords, max_coords))
                mask_ref_crop = mask_ref[slices]
                mask_pred_crop = mask_pred[slices]
                hd_value = calculate_hausdorff_distance(mask_ref_crop, mask_pred_crop)
        
        pad = calculate_percentage_area_difference(mask_ref, mask_pred, ignore_mask)
        
        results['metrics'][r]['HD'] = hd_value
        results['metrics'][r]['PAD'] = pad
    
    # 清理内存
    del seg_ref, seg_pred, mask_ref, mask_pred, union
    return results


def label_or_region_to_key(label_or_region):
    """转换标签为字符串键（nnUNet方式）"""
    return str(label_or_region)


def compute_metrics_on_folder_nnunet_style(folder_ref: str, folder_pred: str, 
                                           regions_or_labels, ignore_label: int = None) -> dict:
    """完全按照nnUNet的方式计算文件夹指标"""
    
    # 获取文件列表
    files_pred = sorted([f for f in os.listdir(folder_pred) if f.endswith('.nii.gz') or f.endswith('.nii')])
    files_ref = sorted([f for f in os.listdir(folder_ref) if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    # 只处理两个文件夹都有的文件
    common_files = list(set(files_pred) & set(files_ref))
    common_files.sort()
    
    print(f"找到 {len(common_files)} 对匹配的文件")
    
    files_ref_full = [os.path.join(folder_ref, i) for i in common_files]
    files_pred_full = [os.path.join(folder_pred, i) for i in common_files]
    
    # 计算每个文件的指标
    process_pair = partial(compute_metrics_nnunet_style, labels_or_regions=regions_or_labels, ignore_label=ignore_label)
    
    num_processes = min(os.cpu_count(), 4)  # 限制进程数以避免内存过载
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_pair, zip(files_ref_full, files_pred_full))
    
    results = [r for r in results if r is not None]
    
    if not results:
        return None
    
    # 计算平均指标（完全按照nnUNet方式）
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            values = [i['metrics'][r][m] for i in results]
            # 处理NaN值和无穷大值
            if m in ['HD']:
                # 对于HD，过滤掉无穷大值
                finite_values = [v for v in values if not np.isinf(v)]
                means[r][m] = np.nanmean(finite_values) if finite_values else float('inf')
            else:
                means[r][m] = np.nanmean(values)

    # 计算前景平均值（完全按照nnUNet方式）
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue  # 跳过背景
            if m in ['HD'] and np.isinf(means[k][m]):
                continue  # 跳过无穷大的HD值
            values.append(means[k][m])
        
        if values:
            foreground_mean[m] = np.mean(values)
        else:
            foreground_mean[m] = float('inf') if m in ['HD'] else np.nan
    
    # 转换结果格式以匹配nnUNet
    def recursive_fix_for_json_export(obj):
        """处理numpy类型以便JSON序列化"""
        if isinstance(obj, dict):
            return {k: recursive_fix_for_json_export(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_fix_for_json_export(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # 应用JSON导出修复
    results = [recursive_fix_for_json_export(i) for i in results]
    means = recursive_fix_for_json_export(means)
    foreground_mean = recursive_fix_for_json_export(foreground_mean)
    
    # 转换键格式以匹配nnUNet输出
    results_converted = deepcopy(results)
    means_converted = {label_or_region_to_key(k): means[k] for k in means.keys()}
    
    # 转换结果中的键
    for i in range(len(results_converted)):
        results_converted[i]['metrics'] = {
            label_or_region_to_key(k): results_converted[i]['metrics'][k]
            for k in results_converted[i]['metrics'].keys()
        }
    
    result = {
        'metric_per_case': results_converted, 
        'mean': means_converted, 
        'foreground_mean': foreground_mean
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='使用nnUNet官方方法计算医学图像分割指标')
    parser.add_argument('--gt', type=str, required=True, 
                       help='Ground Truth目录路径')
    parser.add_argument('--pred', type=str, required=True, 
                       help='预测结果目录路径')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                       help='类别标签列表，例如：--labels 1 2 3')
    parser.add_argument('--ignore_label', type=int, default=None,
                       help='忽略的标签值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.gt):
        print(f"GT目录不存在: {args.gt}")
        return
    
    if not os.path.isdir(args.pred):
        print(f"预测目录不存在: {args.pred}")
        return
    
    print("使用nnUNet官方方法计算指标...")
    result = compute_metrics_on_folder_nnunet_style(
        args.gt, args.pred, args.labels, args.ignore_label
    )
    
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print("计算失败")


def test_calculate_metrics_from_directories(gt_dir, pred_dir, labels=[1, 2], ignore_label=None):
    """测试函数，使用nnUNet官方方法"""
    print("=== 使用nnUNet官方实现方法 ===")
    print(f"标签: {labels}")
    print(f"忽略标签: {ignore_label}")
    
    result = compute_metrics_on_folder_nnunet_style(gt_dir, pred_dir, labels, ignore_label)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    else:
        print("计算失败")
        return None


if __name__ == "__main__":
    # main()
    
    gt_dir = r'nnunetData/nnUNet_raw/Dataset789_ultrasound/mask'
    save_dir = r'evaluation/result'
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    class_list = ["nnUNetTrainer_unetrpp2"]
    
    for class_name in class_list:
        pred_dir = os.path.join(r'evaluation/pred', class_name)
        print(pred_dir)
        result = test_calculate_metrics_from_directories(gt_dir, pred_dir, labels=[1, 2], ignore_label=0)
        if result is not None:
            save_path = os.path.join(save_dir, class_name + 'quick.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {save_path}")