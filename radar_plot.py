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
    labels = ['1/RMSE', 'SNR', '1/NMSE', 'MI', 'Average']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    # 创建三个垂直子图，调整大小为 7x14
    fig, axs = plt.subplots(3, 1, figsize=(7, 14), subplot_kw=dict(polar=True))

    # 定义子图标题和叠放顺序
    titles = ["Single Modality", "Dual Modality", "Triple Modality"]
    modality_data = [metrics_single, metrics_dual, metrics_triple]
    modality_labels = [
        ["PSD", "WPD", "TSE"],  # 单模态中 TSE 放最上层
        ["TSE + PSD", "PSD + WPD", "TSE + WPD"],  # 双模态中 TSE + WPD 放最上层
        ["TSE + PSD + WPD"]  # 三模态中唯一数据项
    ]

    # 绘制每个模态的雷达图
    for ax, title, data, method_labels in zip(axs, titles, modality_data, modality_labels):
        ax.set_ylim(0, 1)  # 统一范围
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)  # 设置角度标签
        ax.set_title(title, size=14, y=1.1)

        # 根据所需顺序绘制
        for method, values in zip(method_labels, data):
            values = np.append(values, values[0])  # 闭合数据点
            ax.plot(angles, values, linewidth=1.5, label=method)
            ax.fill(angles, values, alpha=0.25)

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)

        # 保留 Y 轴刻度线但去掉刻度数字
        ax.yaxis.set_tick_params(labelleft=False)  # 去掉左侧的刻度数字

    plt.tight_layout()
    plt.show()

# 定义数据（所有单、双、三模态的数据）
# metrics_all = {
#     "PSD": [2.95, 15.63, 0.0663, 3.8275],
#     "WPD": [3.13, 14.545, 0.0698, 3.8063],
#     "TSE": [3.28, 14.2687, 0.0686, 3.31],
#     "TSE + PSD": [2.93, 15.655, 0.0661, 3.825],
#     "PSD + WPD": [2.95, 15.5575, 0.066, 3.83],
#     "TSE + WPD": [3.00, 15.405, 0.0687, 3.8188],
#     "TSE + PSD + WPD": [2.91, 15.6788, 0.0649, 3.85]
# }
#
# plot_modality_radar(metrics_all)
