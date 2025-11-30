#!/usr/bin/env python3
"""
run_plot.py - 自动化超参数敏感度分析与消融实验脚本
基于 main.py 的逻辑进行封装，针对 OLMA 算法进行深度测试。

输出:
 1. plots/sensitivity/*.png (敏感度曲线)
 2. logs/ablation_results.csv (消融实验数据)
"""

import os
import time
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入 main.py 中的工具函数，确保逻辑一致性
# 注意：main.py 必须在同一目录下
import main

# -----------------------------
# 配置
# -----------------------------
TARGET_SOLVER = "solvers.OLMA_Solver_perfect.OLMA_Solver"
PLOT_DIR = "plots/sensitivity"
LOG_DIR = "logs"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 默认基础配置
BASE_ENV_CFG = {
    "F_max": 2.0,  # GHz
    "B_max": 20.0,  # MHz
    "P_max": 1.0,  # Watt
    "weights": [0.5, 0.3, 0.2]  # [We, Wc, Wh] (示例)
}

BASE_SOLVER_CFG = {
    "V": 50.0,  # Lyapunov parameter
    "I_bcd": 10,  # BCD iterations
    "I_sca": 5,  # SCA iterations
    "epsilon": 1e-3,  # Convergence threshold
    "ablation": "none"  # none, no_power, no_bw, etc.
}


# -----------------------------
# 核心运行函数
# -----------------------------
def run_single_experiment(env_config_update, solver_config_update, run_name):
    """
    运行单次实验并返回关键指标 summary
    """
    # 1. 合并配置
    env_cfg = BASE_ENV_CFG.copy()
    env_cfg.update(env_config_update)

    solver_cfg = BASE_SOLVER_CFG.copy()
    solver_cfg.update(solver_config_update)

    # 2. 初始化环境 (动态加载，与 main.py 保持一致)
    try:
        env_module = importlib.import_module("solvers.environment")
        VEC_Environment = getattr(env_module, "VEC_Environment")
        env = VEC_Environment(env_cfg)
        env.reset()
    except Exception as e:
        print(f"[{run_name}] 环境加载失败: {e}")
        return None

    # 3. 初始化求解器
    try:
        solver_obj = main.load_solver(TARGET_SOLVER, env_cfg, solver_cfg)
    except Exception as e:
        print(f"[{run_name}] 求解器加载失败: {e}")
        return None

    # 4. 运行循环 (使用 main.SLOTS)
    # 不使用 logger 保存文件以加快速度，仅在内存统计
    metrics = {
        "costs": [],
        "delays": [],
        "energies": []
    }

    # print(f"Running {run_name}...", end="", flush=True)

    for t in range(main.SLOTS):
        state = env.get_state()
        decision = main.timed_solve(solver_obj, state)
        diag = env.step(decision, state)

        # 收集关键指标
        metrics["costs"].append(diag.get("total_cost", 0.0))

        d_total = diag.get("delay_queue", 0) + diag.get("delay_tx", 0) + \
                  diag.get("delay_proc", 0) + diag.get("delay_backhaul", 0)
        metrics["delays"].append(d_total)

        e_total = diag.get("energy_tx", 0) + diag.get("energy_srv", 0)
        metrics["energies"].append(e_total)

    # print(" Done.")

    # 5. 计算统计值
    summary = {
        "C_mean": np.mean(metrics["costs"]),
        "Delay_mean": np.mean(metrics["delays"]),
        "Energy_mean": np.mean(metrics["energies"]),
    }
    return summary

# -----------------------------
# 模块 1: 敏感度分析
# -----------------------------
def run_sensitivity_analysis():
    print("\n" + "=" * 50)
    print(">>> 开始超参数敏感度分析 (Sensitivity Analysis)")
    print("=" * 50)

    # 定义要扫描的参数及其范围 (使用您要求的 N >= 20 的配置)
    experiments = [
        # 1. V: 成本-延迟权衡 - 21 个点
        ("V (Lyapunov)", "solver", "V",
         np.linspace(0, 400, 21).round(1).tolist(),
         "V Parameter (Lyapunov Control)"),

        # 2. 物理限制 (Physical Limits) - 20 个点
        ("Pmax (Power Limit)", "env", "P_max",
         np.linspace(0.1, 2.0, 20).round(2).tolist(),
         "Power Limit (W)"),

        ("Fmax (CPU Freq)", "env", "F_max",
         np.linspace(1.0, 3.0, 20).round(2).tolist(),
         "Max Frequency (GHz)"),

        # 3. 算法参数 (Algorithm) - BCD 20 个点
        ("I_bcd (Iterations)", "solver", "I_bcd",
         list(range(1, 21)),
         "BCD Iterations"),

        # 4. Epsilon (Convergence) - 10 个对数点
        ("Epsilon (Convergence)", "solver", "epsilon",
         np.logspace(-5, -2, 10).round(6).tolist(),
         "Epsilon (Convergence Threshold)"),

        # 5. 权重 (Weights) - 20 个点
        ("Weight_Energy", "env", "weights_E",
         np.linspace(0.05, 0.95, 20).round(3).tolist(),
         "Energy Weight ($W_E$)")
    ]

    for title, cfg_type, key, values, xlabel in experiments:
        # 初始化 results 列表
        results = []
        print(f"\n--- Testing Sensitivity: {title} ({len(values)} points) ---")

        for v in values:
            env_upd = {}
            slv_upd = {}

            # 特殊处理权重
            if key == "weights_E":
                remain = 1.0 - v
                w_c = remain / 2
                w_h = remain / 2
                env_upd["weights"] = [v, w_c, w_h]
            else:
                if cfg_type == "env":
                    env_upd[key] = v
                else:
                    slv_upd[key] = v

            # 运行实验并收集结果
            summary = run_single_experiment(env_upd, slv_upd, f"{key}={v}")
            if summary:
                results.append(summary["C_mean"])
            else:
                # 如果运行失败，可以添加一个 NaN 或 0.0 来保持列表长度一致
                results.append(0.0)

                # 绘图 - 此时 results 已经被定义和填充
        plt.figure(figsize=(7, 5))
        # 绘图代码可以正常运行
        plt.plot(values, results, marker='o', linewidth=2, color='#1f77b4', label='Avg Cost')
        plt.title(f"Sensitivity Analysis: {title}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Average System Cost", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 标记扫描点的数量
        plt.text(0.98, 0.02, f"N={len(values)}", transform=plt.gca().transAxes,
                 fontsize=9, color='gray', ha='right')

        # 如果是 epsilon，使用对数坐标
        if key == "epsilon":
            plt.xscale('log')

        filename = f"sensitivity_{key}.png"
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plot: {filename}")


# -----------------------------
# 模块 2: 消融实验 (CSV)
# -----------------------------
def run_ablation_study():
    print("\n" + "=" * 50)
    print(">>> 开始消融实验 (Ablation Study)")
    print("=" * 50)

    # 定义消融变体
    # 这里假设 solver 内部会根据 'ablation' 字段跳过某些优化步骤
    variants = [
        ("OLMA (Full)", "none"),
        ("w/o Power Control", "no_power"),
        ("w/o Bandwidth Alloc", "no_bw"),
        ("w/o Computation Offloading", "no_offload"),  # 强制本地或全卸载
        ("w/o Freq Scaling", "no_freq")
    ]

    records = []

    for label, mode in variants:
        print(f"Running Ablation: {label} ...")

        # 只需要修改 solver 的 ablation 参数
        slv_upd = {"ablation": mode}
        summary = run_single_experiment({}, slv_upd, label)

        if summary:
            rec = {
                "Method": label,
                "Avg Cost": summary["C_mean"],
                "Avg Delay": summary["Delay_mean"],
                "Avg Energy": summary["Energy_mean"]
            }
            records.append(rec)

    # 保存 CSV
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(LOG_DIR, "ablation_results.csv")
        df.to_csv(csv_path, index=False)
        print("\n" + "-" * 50)
        print(f"消融实验结果已保存至: {csv_path}")
        print("-" * 50)
        print(df.to_string())
    else:
        print("消融实验未产生数据。")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 设置绘图风格
    plt.style.use('default')
    # 尝试设置字体以支持中文（如果在中文环境下），可选
    plt.rcParams['axes.unicode_minus'] = False

    print(">>> Loading environment and solver module from main.py context...")

    # 1. 运行敏感度分析
    run_sensitivity_analysis()

    # 2. 运行消融实验
    run_ablation_study()

    print("\n>>> All analysis tasks completed.")
