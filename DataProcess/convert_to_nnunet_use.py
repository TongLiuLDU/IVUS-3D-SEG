import os
import shutil
from typing import List, Tuple


SRC_BASE = "nnunetData/IVUS-3D-SEG"
SPLIT_DIR = "DataProcess/result/split_result"
DST_BASE = "nnunetData/nnUNet_raw/Dataset789_ultrasound"
# 若为 True，则在复制前清空目标子目录，避免历史残留文件导致数量偏差
CLEAN_TARGET = True


def read_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def find_image_and_mask(case_name: str) -> Tuple[str, str]:
    case_dir = os.path.join(SRC_BASE, case_name)
    # 优先使用 .nii.gz
    img_nii_gz = os.path.join(case_dir, f"{case_name}_image.nii.gz")
    img_nii = os.path.join(case_dir, f"{case_name}_image.nii")
    msk_nii_gz = os.path.join(case_dir, f"{case_name}_mask.nii.gz")
    msk_nii = os.path.join(case_dir, f"{case_name}_mask.nii")

    img = img_nii_gz if os.path.exists(img_nii_gz) else (img_nii if os.path.exists(img_nii) else "")
    msk = msk_nii_gz if os.path.exists(msk_nii_gz) else (msk_nii if os.path.exists(msk_nii) else "")
    return img, msk


def copy_case(case_name: str, dst_dir: str, src_path: str) -> bool:
    if not src_path or not os.path.exists(src_path):
        return False
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return True


def main():
    imagesTr = os.path.join(DST_BASE, "imagesTr")
    imagesTs = os.path.join(DST_BASE, "imagesTs")
    labelsTr = os.path.join(DST_BASE, "labelsTr")
    masksTs = os.path.join(DST_BASE, "mask")
    for d in [imagesTr, imagesTs, labelsTr, masksTs]:
        ensure_dir(d)

    # 可选：清空目标目录，避免旧文件残留
    if CLEAN_TARGET:
        for d in [imagesTr, imagesTs, labelsTr, masksTs]:
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if os.path.isfile(path):
                    os.remove(path)

    train_cases = read_list(os.path.join(SPLIT_DIR, "train_cases.txt"))
    val_cases = read_list(os.path.join(SPLIT_DIR, "val_cases.txt"))
    test_cases = read_list(os.path.join(SPLIT_DIR, "test_cases.txt"))

    # 去重并固定顺序，避免重复复制导致数量异常
    # imagesTr 需要 训练集 + 验证集 的 image
    imagesTr_cases = unique_keep_order(train_cases + val_cases)
    imagesTs_cases = unique_keep_order(test_cases)
    labelsTr_cases = unique_keep_order(train_cases + val_cases)
    maskTs_cases = unique_keep_order(test_cases)

    report_path = os.path.join(DST_BASE, "copy_report.txt")
    copied = []
    missing = []

    # 训练集与验证集的 image -> imagesTr
    for c in imagesTr_cases:
        img, _ = find_image_and_mask(c)
        ok = copy_case(c, imagesTr, img)
        if ok:
            copied.append(f"imagesTr: {c} -> {os.path.basename(img)}")
        else:
            missing.append(f"imagesTr missing image: {c}")

    # 测试集 image -> imagesTs
    for c in imagesTs_cases:
        img, _ = find_image_and_mask(c)
        ok = copy_case(c, imagesTs, img)
        if ok:
            copied.append(f"imagesTs: {c} -> {os.path.basename(img)}")
        else:
            missing.append(f"imagesTs missing image: {c}")

    # 训练集与验证集的 mask -> labelsTr
    for c in labelsTr_cases:
        _, msk = find_image_and_mask(c)
        ok = copy_case(c, labelsTr, msk)
        if ok:
            copied.append(f"labelsTr: {c} -> {os.path.basename(msk)}")
        else:
            missing.append(f"labelsTr missing mask: {c}")

    # 测试集的 mask -> mask
    for c in maskTs_cases:
        _, msk = find_image_and_mask(c)
        ok = copy_case(c, masksTs, msk)
        if ok:
            copied.append(f"maskTs: {c} -> {os.path.basename(msk)}")
        else:
            missing.append(f"maskTs missing mask: {c}")
    
    # 写报告与校验
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("拷贝报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"期望 imagesTr = train({len(train_cases)}) + val({len(val_cases)}) = {len(train_cases) + len(val_cases)}\n")
        f.write(f"期望 imagesTs = test({len(test_cases)})\n")
        f.write(f"期望 labelsTr = train({len(train_cases)}) + val({len(val_cases)}) = {len(train_cases) + len(val_cases)}\n")
        f.write(f"期望 mask   = test({len(test_cases)})\n\n")

        # 实际数量
        f.write("实际数量:\n")
        f.write(f"imagesTr: {len(os.listdir(imagesTr))} (按去重case={len(imagesTr_cases)})\n")
        f.write(f"imagesTs: {len(os.listdir(imagesTs))} (按去重case={len(imagesTs_cases)})\n")
        f.write(f"labelsTr: {len(os.listdir(labelsTr))} (按去重case={len(labelsTr_cases)})\n")
        f.write(f"mask:     {len(os.listdir(masksTs))} (按去重case={len(maskTs_cases)})\n\n")

        if copied:
            f.write("成功拷贝:\n")
            for line in copied:
                f.write(line + "\n")
            f.write("\n")

        if missing:
            f.write("缺失文件:\n")
            for line in missing:
                f.write(line + "\n")

    print("完成. 明细见:", report_path)



def rename():
    # 对imagesTr和labelsTr中的文件进行重命名，1-46
    imagesTr_dir = os.path.join(DST_BASE, "imagesTr")
    labelsTr_dir = os.path.join(DST_BASE, "labelsTr")
    for i, file in enumerate(sorted(os.listdir(imagesTr_dir))):
        if file.endswith(".nii.gz"):
            if i+1<10:
                new_name = file.replace("_image.nii.gz", f"_00{i+1}_0000.nii.gz")
            else:
                new_name = file.replace("_image.nii.gz", f"_0{i+1}_0000.nii.gz")
            os.rename(os.path.join(imagesTr_dir, file), os.path.join(imagesTr_dir, new_name))
            # print(new_name)
            
    for i, file in enumerate(sorted(os.listdir(labelsTr_dir))):
        if file.endswith(".nii.gz"):
            if i+1<10:
                new_name = file.replace("_mask.nii.gz", f"_00{i+1}.nii.gz")
            else:
                new_name = file.replace("_mask.nii.gz", f"_0{i+1}.nii.gz")
            os.rename(os.path.join(labelsTr_dir, file), os.path.join(labelsTr_dir, new_name))
            # print(new_name)
            
    # 对imagesTs和masksTs中的文件进行重命名，47-75
    imagesTs_dir = os.path.join(DST_BASE, "imagesTs")
    masksTs_dir = os.path.join(DST_BASE, "mask")
    for i, file in enumerate(sorted(os.listdir(imagesTs_dir))):
        if file.endswith(".nii.gz"):
            if i+47<10:
                new_name = file.replace("_image.nii.gz", f"_00{i+47}_0000.nii.gz")
            else:
                new_name = file.replace("_image.nii.gz", f"_0{i+47}_0000.nii.gz")
            os.rename(os.path.join(imagesTs_dir, file), os.path.join(imagesTs_dir, new_name))

    for i, file in enumerate(sorted(os.listdir(masksTs_dir))):
        if file.endswith(".nii.gz"):
            if i+47<10:
                new_name = file.replace("_mask.nii.gz", f"_00{i+47}.nii.gz")
            else:
                new_name = file.replace("_mask.nii.gz", f"_0{i+47}.nii.gz")
            os.rename(os.path.join(masksTs_dir, file), os.path.join(masksTs_dir, new_name))

            
main()            
rename()