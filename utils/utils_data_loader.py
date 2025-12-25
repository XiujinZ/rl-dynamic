# utils_data_loader.py
import os
import numpy as np
import glob

def load_od_matrices(od_dir: str) -> np.ndarray:
    """
    从文件夹加载所有 OD 概率矩阵并按时间排序堆叠成 [T_total, N, N]
    """
    paths = sorted(glob.glob(os.path.join(od_dir, "OD_prob_day*_hour*.npy")))
    print(f"[INFO] 共找到 {len(paths)} 个 OD 文件。")

    mats = [np.load(p) for p in paths]
    od_mats = np.stack(mats, axis=0)  # [T_total, N, N]
    print(f"[INFO] 载入 OD 矩阵 shape = {od_mats.shape}")
    return od_mats


def load_residents(path: str) -> np.ndarray:
    residents = np.load(path)
    print(f"[INFO] residents shape = {residents.shape}, 总人口 = {residents.sum():.0f}")
    return residents
