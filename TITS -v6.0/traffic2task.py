#!/usr/bin/env python3
"""
traffic2task.py - 将真实交通轨迹 CSV 转换为 OLMA 可用的数据集
输出：real_dataset_converted.csv
格式包含 Din, Dout, Cin, Channel (MB/cycles/channel Gain)
"""

import pandas as pd
import numpy as np

# === 用户输入：请替换为你的真实 CSV 文件名 ===
INPUT_CSV = "real_datasets.csv"
OUTPUT_CSV = "real_dataset_converted.csv"

# === 基站位置（可改，不改也行） ===
BS_X, BS_Y = 2230500, 1375500

def convert():
    df = pd.read_csv(INPUT_CSV)

    # 检查列
    required_cols = ["Global_X", "Global_Y", "v_Vel", "v_Acc"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"缺失列: {c}")

    # === 1. Din 生成 (MB) ===
    # 摄像头任务模型：速度越快，图像复杂度越高
    df["Din"] = 0.5 + 0.02 * df["v_Vel"]        # MB

    # === 2. Dout 生成 (MB) ===
    df["Dout"] = df["Din"] * 0.1                # 下行更小

    # === 3. Cin 生成 (cycles) ===
    df["Cin"] = (2e8 + 1e7 * np.abs(df["v_Acc"]) + 5e6 * df["v_Vel"]).astype(float)

    # === 4. Channel 生成 ===
    dx = df["Global_X"] - BS_X
    dy = df["Global_Y"] - BS_Y
    d = np.sqrt(dx*dx + dy*dy)

    pathloss = (1 / (d + 1)) ** 3
    shadow = np.random.lognormal(mean=0, sigma=0.4, size=len(df))
    df["Channel"] = pathloss * shadow

    # === 导出任务数据集 ===
    out = df[["Din", "Dout", "Cin", "Channel"]]
    out.to_csv(OUTPUT_CSV, index=False)

    print("转换成功！生成任务数据集:", OUTPUT_CSV)
    print(out.head())


if __name__ == "__main__":
    convert()