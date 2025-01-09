import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 使用 TKAgg 后端
matplotlib.use('TKAgg')


def compare_metrics_radar(rmse_clean, rmse_normal_asr, rmse_picard, rmse_SSP,
                          nmse_clean, nmse_normal_asr, nmse_picard, nmse_SSP,
                          snr_clean, snr_normal_asr, snr_picard, snr_SSP,
                          mi_clean, mi_normal_asr, mi_picard, mi_SSP):
    """
    绘制四种方法的雷达图，表示五个指标（1/RMSE、1/NMSE、SNR、MI、平均值）。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 计算每种方法的平均值并将其作为第五个指标
    methods_metrics = {
        'MASR': [1 / rmse_clean, 1 / nmse_clean, snr_clean, mi_clean],
        'ASR': [1 / rmse_normal_asr, 1 / nmse_normal_asr, snr_normal_asr, mi_normal_asr],
        'SSP': [1 / rmse_SSP, 1 / nmse_SSP, snr_SSP, mi_SSP],
        'Picard': [1 / rmse_picard, 1 / nmse_picard, snr_picard, mi_picard],

    }

    for method, metrics in methods_metrics.items():
        average_metric = np.mean(metrics)
        metrics.append(average_metric)

    # 2. 数据归一化并映射到 [0.2, 0.8] 区间
    metrics = np.array(list(methods_metrics.values()))
    min_value = metrics.min(axis=0)
    max_value = metrics.max(axis=0)
    norm_metrics = (metrics - min_value) / (max_value - min_value + 1e-10)  # 归一化到 [0, 1]

    # 非线性映射：增加中心分离效果（指数映射）
    norm_metrics = norm_metrics ** 0.5  # 调整参数以更灵活地控制内外比例
    norm_metrics = norm_metrics * 0.6 + 0.2  # 再次映射到 [0.2, 0.8] 区间

    # 3. 设置雷达图参数
    labels = ['1/RMSE', '1/NMSE', 'SNR', 'MI', 'Average']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 添加第一个角度到最后以闭合

    norm_metrics = np.concatenate((norm_metrics, norm_metrics[:, [0]]), axis=1)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # 4. 颜色定义（深色在下，浅色在上）

    # 5. 绘制每种方法的雷达图线
    for i, (method, values) in enumerate(methods_metrics.items()):
        ax.plot(angles, norm_metrics[i], linewidth=1.5, label=method)
        ax.fill(angles, norm_metrics[i], alpha=0.5)

    # 6. 设置径向范围并去掉径向刻度标签
    ax.set_ylim(0, 1)  # 设置范围从0到1
    ax.set_yticklabels([])  # 移除径向刻度数字

    # 设置环绕标签布局并调整标签位置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, ha='center', va='top', fontweight='bold', fontsize=12)

    ax.legend(loc='best')

    # 显示图形
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_modality_radar(metrics_all):
    """
    绘制单模态、双模态和三模态的雷达图。

    参数:
    metrics_all (dict): 各模态的指标数据字典，键为模态名称，值为列表形式的指标数据。
    """
    # 添加平均值作为第五个指标
    for method, metrics in metrics_all.items():
        average_metric = np.mean(metrics)
        metrics.append(average_metric)

    # 提取 RMSE 和 NMSE 取倒数处理，并重新组织数据
    processed_metrics_all = {}
    for modality, metrics in metrics_all.items():
        processed_metrics_all[modality] = [
            1 / metrics[0],  # 1/RMSE
            metrics[1],  # SNR
            1 / metrics[2],  # 1/NMSE
            metrics[3],  # MI
            metrics[4]  # Average
        ]

    # 数据转换为 NumPy 数组，统一归一化
    metrics_array = np.array(list(processed_metrics_all.values()))
    min_val, max_val = metrics_array.min(axis=0), metrics_array.max(axis=0)
    norm_metrics_all = (metrics_array - min_val) / (max_val - min_val + 1e-10) * 0.5 + 0.3  # 映射到 [0.2, 0.8]

    # 分离单、双、三模态数据
    metrics_single = norm_metrics_all[:3]
    metrics_dual = norm_metrics_all[3:6]
    metrics_triple = norm_metrics_all[6:]

    # 设定雷达图的标签和角度
    labels = ['1/RMSE', 'SAR', '1/NMSE', 'MI', 'Average']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    # 创建 2x8 网格布局
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 8, figure=fig)

    # 合并第一行格子 1-4 和 5-8
    ax1 = fig.add_subplot(gs[0, 0:4], polar=True)  # 单模态
    ax2 = fig.add_subplot(gs[0, 4:8], polar=True)  # 双模态

    # 合并第二行格子 3-6
    ax3 = fig.add_subplot(gs[1, 2:6], polar=True)  # 三模态，合并格子3-6

    # 绘制每个模态的雷达图
    # 绘制单模态图
    ax1.set_ylim(0, 1)  # 统一范围
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels)
    ax1.set_title("Single Modality", size=14, y=1.1)
    # 移动标签向外
    for label in ax1.get_xticklabels():
        x_pos, y_pos = label.get_position()  # 获取原始位置
        # 将标签位置的y坐标放大，使其远离中心
        label.set_position((x_pos, y_pos-0.05))  # 可以调整倍数 1.2 来控制距离
    for method, values in zip(["PSD", "WPD", "TSE"], metrics_single):
        values = np.append(values, values[0])  # 闭合数据点
        ax1.plot(angles, values, linewidth=1.5, label=method)
        ax1.fill(angles, values, alpha=0.25)

    # 绘制双模态图
    ax2.set_ylim(0, 1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels)
    ax2.set_title("Dual Modality", size=14, y=1.1)
    for label in ax2.get_xticklabels():
        x_pos, y_pos = label.get_position()  # 获取原始位置
        # 将标签位置的y坐标放大，使其远离中心
        label.set_position((x_pos, y_pos-0.05))  # 可以调整倍数 1.2 来控制距离
    for method, values in zip(["TSE + PSD", "PSD + WPD", "TSE + WPD"], metrics_dual):
        values = np.append(values, values[0])
        ax2.plot(angles, values, linewidth=1.5, label=method)
        ax2.fill(angles, values, alpha=0.25)

    # 绘制三模态图
    ax3.set_ylim(0, 1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    ax3.set_title("Triple Modality", size=14, y=1.1)
    for label in ax3.get_xticklabels():
        x_pos, y_pos = label.get_position()  # 获取原始位置
        # 将标签位置的y坐标放大，使其远离中心
        label.set_position((x_pos, y_pos-0.05))  # 可以调整倍数 1.2 来控制距离
    for method, values in zip(["TSE + PSD + WPD"], metrics_triple):
        values = np.append(values, values[0])
        ax3.plot(angles, values, linewidth=1.5, label=method)
        ax3.fill(angles, values, alpha=0.25)

    # 设置所有子图的图例
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.33, 1.1), fontsize=8)

    # 保留 Y 轴刻度线但去掉刻度数字
    ax1.yaxis.set_tick_params(labelleft=False)
    ax2.yaxis.set_tick_params(labelleft=False)
    ax3.yaxis.set_tick_params(labelleft=False)

    # 调整布局，增加紧凑感
    plt.tight_layout()
    plt.show()



# 定义数据（所有单、双、三模态的数据）
metrics_all = {
    "PSD": [2.95, 15.63, 0.0663, 3.8275],
    "WPD": [3.13, 14.545, 0.0698, 3.8063],
    "TSE": [3.28, 14.2687, 0.0686, 3.31],
    "TSE + PSD": [2.93, 15.655, 0.0661, 3.825],
    "PSD + WPD": [2.95, 15.5575, 0.066, 3.83],
    "TSE + WPD": [3.00, 15.405, 0.0687, 3.8188],
    "TSE + PSD + WPD": [2.91, 15.6788, 0.0649, 3.85]
}

plot_modality_radar(metrics_all)
