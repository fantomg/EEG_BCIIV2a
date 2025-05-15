import mne
from matplotlib import pyplot as plt
from mne import channel_type
import numpy as np
from mne.utils.check import _check_option
from mne.viz.evoked import _plot_evoked
from mne.viz.topomap import _set_contour_locator
from mne.viz.utils import _check_time_unit, _validate_if_list_of_axes, _process_times, plt_show, _setup_vmin_vmax


def plot_waveform_differences(epochs, cleaned_avg1, cleaned_avg2, cleaned_avg3, cleaned_avg4, channel_names):
    """
    绘制原始信号与不同清理方法的波形差异，用于适应双栏论文格式。

    参数:
    - epochs: 原始的 Epochs 对象
    - cleaned_avg1, cleaned_avg2, cleaned_avg3, cleaned_avg4: 分别对应 MASR, ASR, Picard-ICA 和 SSP 清理后的平均值
    - channel_names: 包含要绘制的通道名称的列表

    返回:
    - None
    """
    # 加载标准通道位置
    montage = mne.channels.make_standard_montage('standard_1020')

    # 原始数据的平均 evoked
    evoked_original = epochs.average(picks=channel_names)
    evoked_original.set_montage(montage)

    # 生成清理后 evoked 数据并设置蒙太奇
    cleaned_avgs = [cleaned_avg3, cleaned_avg4, cleaned_avg2, cleaned_avg1]  # 调整顺序
    evoked_cleaned = []
    for cleaned in cleaned_avgs:
        evoked = cleaned.average()
        evoked.set_montage(montage)
        evoked_cleaned.append(evoked)

    # 计算波形差异
    differences = [mne.combine_evoked([evoked_original, evoked], weights=[1, -1]) for evoked in evoked_cleaned]

    # 设置画布大小和子图布局 (适应双栏宽度)
    fig, axs = plt.subplots(4, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # 定义标题列表
    titles = ["Picard-ICA", "SSP", "ASR", "MASR"]  # 调整标题顺序

    # 绘制每个方法的差异波形图并添加标题
    for ax, diff_evoked, title in zip(axs, differences, titles):
        diff_evoked.plot(spatial_colors=True, axes=ax, show=False)
        ax.set_title(title, fontsize=10)  # 设置方法名称作为标题
        ax.tick_params(axis='both', which='major', labelsize=6)  # 调整刻度标签字体大小

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.3)  # 减小子图之间的垂直间距
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 顶部保留少量空间
    plt.show()


def plot_waveform_before_after(epochs, cleaned_avg1, cleaned_avg2, cleaned_avg3, cleaned_avg4, channel_names):
    """
    绘制原始信号与不同清理方法处理后的波形。

    参数:
    - epochs: 原始的 Epochs 对象
    - cleaned_avg1, cleaned_avg2, cleaned_avg3, cleaned_avg4: 分别对应 MASR, ASR, Picard-ICA 和 SSP 清理后的平均值
    - channel_names: 包含要绘制的通道名称的列表

    返回:
    - None
    """
    # 加载标准通道位置
    montage = mne.channels.make_standard_montage('standard_1020')

    # 原始数据的平均 evoked
    evoked_original = epochs.average(picks=channel_names)
    evoked_original.set_montage(montage)

    # 生成清理后 evoked 数据并设置蒙太奇
    cleaned_avgs = [cleaned_avg3, cleaned_avg4, cleaned_avg2, cleaned_avg1]  # 处理后的清理方法
    evoked_cleaned = []
    for cleaned in cleaned_avgs:
        cleaned.average().plot_joint()
        evoked = cleaned.average()
        evoked.set_montage(montage)
        evoked_cleaned.append(evoked)

    # 设置画布大小和子图布局 (适应双栏宽度)
    fig, axs = plt.subplots(5, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

    # 定义标题列表
    titles = ["Raw Signal", "Picard-ICA", "SSP", "ASR", "MASR"]  # 子图标题

    # 绘制原始信号
    evoked_original.plot(spatial_colors=True, axes=axs[0], show=False)
    axs[0].set_title(titles[0], fontsize=10)  # 原始信号的标题
    axs[0].tick_params(axis='both', which='major', labelsize=6)

    # 绘制每个处理方法的波形图并添加标题
    for ax, evoked, title in zip(axs[1:], evoked_cleaned, titles[1:]):
        evoked.plot(spatial_colors=True, axes=ax, show=False)  # 绘制处理后的波形
        ax.set_title(title, fontsize=10)  # 设置方法名称作为标题
        ax.tick_params(axis='both', which='major', labelsize=6)  # 调整刻度标签字体大小

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.2)  # 减小子图之间的垂直间距
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 顶部保留少量空间
    plt.show()


def prepare_joint_axes(n_maps, figsize=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(3, n_maps, height_ratios=[2, 2, 2], figure=fig)
    map_ax = [fig.add_subplot(gs[0, x]) for x in range(n_maps)]  # first row
    main_ax = fig.add_subplot(gs[1, :])  # second row
    diff_ax = fig.add_subplot(gs[2, :])  # third row
    return fig, main_ax, map_ax, diff_ax


def plot_evoked_joint(
        evoked,
        raw,
        times="peaks",
        title="",
        picks=None,
        exclude=None,
        show=True,
        ts_args=None,
        topomap_args=None,
):
    from matplotlib.patches import ConnectionPatch

    # 设置通道位置
    montage = mne.channels.make_standard_montage('standard_1020')

    # 原始数据的平均 evoked
    evoked_original = raw.average(picks=picks)
    evoked_original.set_montage(montage)

    # 计算每种方法的 evoked 并得到差异
    evoked_cleaned = evoked.average().set_montage(montage)

    # prepare axes for topomap
    if ts_args is not None and not isinstance(ts_args, dict):
        raise TypeError(f"ts_args must be dict or None, got type {type(ts_args)}")
    ts_args = dict() if ts_args is None else ts_args.copy()
    ts_args["time_unit"], _ = _check_time_unit(
        ts_args.get("time_unit", "s"), evoked.times
    )
    topomap_args = dict() if topomap_args is None else topomap_args.copy()

    got_axes = False
    illegal_args = {"show", "times", "exclude"}
    for args in (ts_args, topomap_args):
        if any(x in args for x in illegal_args):
            raise ValueError(
                "Don't pass any of {} as *_args.".format(", ".join(list(illegal_args)))
            )
    if ("axes" in ts_args) or ("axes" in topomap_args):
        if not (("axes" in ts_args) and ("axes" in topomap_args)):
            raise ValueError(
                "If one of `ts_args` and `topomap_args` contains "
                "'axes', the other must, too."
            )
        _validate_if_list_of_axes([ts_args["axes"]], 1)

        if times in (None, "peaks"):
            n_topomaps = 3 + 1
        else:
            assert not isinstance(times, str)
            n_topomaps = len(times) + 1

        _validate_if_list_of_axes(list(topomap_args["axes"]), n_topomaps)
        got_axes = True

    # channel selection
    # simply create a new evoked object with the desired channel selection
    # Need to deal with proj before picking to avoid bad projections
    proj = topomap_args.get("proj", True)
    proj_ts = ts_args.get("proj", True)
    if proj_ts != proj:
        raise ValueError(
            f'topomap_args["proj"] (default True, got {proj}) must match '
            f'ts_args["proj"] (default True, got {proj_ts})'
        )
    _check_option('topomap_args["proj"]', proj, (True, False, "reconstruct"))
    evoked = evoked.copy()
    if proj:
        evoked.apply_proj()
        if proj == "reconstruct":
            evoked._reconstruct_proj()
    topomap_args["proj"] = ts_args["proj"] = False  # don't reapply
    evoked.pick(picks, exclude=exclude)
    info = evoked.info
    ch_types = info.get_channel_types(unique=True, only_data_chs=True)

    # if multiple sensor types: one plot per channel type, recursive call
    if len(ch_types) > 1:
        if got_axes:
            raise NotImplementedError(
                "Currently, passing axes manually (via `ts_args` or "
                "`topomap_args`) is not supported for multiple channel types."
            )
        figs = list()
        for this_type in ch_types:  # pick only the corresponding channel type
            ev_ = evoked.copy().pick(
                [
                    info["ch_names"][idx]
                    for idx in range(info["nchan"])
                    if channel_type(info, idx) == this_type
                ]
            )
            if len(ev_.info.get_channel_types(unique=True)) > 1:
                raise RuntimeError(
                    "Possibly infinite loop due to channel "
                    "selection problem. This should never "
                    "happen! Please check your channel types."
                )
            figs.append(
                plot_evoked_joint(
                    ev_,
                    times=times,
                    title=title,
                    show=show,
                    ts_args=ts_args,
                    exclude=list(),
                    topomap_args=topomap_args,
                )
            )
        return figs

    # set up time points to show topomaps for
    times_sec = _process_times(evoked_cleaned, times, few=True)
    del times
    _, times_ts = _check_time_unit(ts_args["time_unit"], times_sec)
    if not got_axes:
        fig, ts_ax, map_ax, diff_ax = prepare_joint_axes(len(times_sec), figsize=(7, 6))
        cbar_ax = None
    else:
        ts_ax = ts_args["axes"]
        del ts_args["axes"]
        map_ax = topomap_args["axes"][:-1]
        cbar_ax = topomap_args["axes"][-1]
        del topomap_args["axes"]
        fig = cbar_ax.figure

        # butterfly/time series plot
        # most of this code is about passing defaults on demand
    ts_args_def = dict(
        picks=None,
        unit=True,
        ylim=None,
        xlim="tight",
        proj=False,
        hline=None,
        units=None,
        scalings=None,
        titles=None,
        gfp=False,
        window_title=None,
        spatial_colors=True,
        zorder="std",
        sphere=None,
        draw=False,
    )
    ts_args_def.update(ts_args)
    _plot_evoked(
        evoked_cleaned, axes=ts_ax, show=False, plot_type="butterfly", exclude=[], **ts_args_def
    )
    diff = mne.combine_evoked([evoked_original, evoked_cleaned], weights=[1, -1])
    _plot_evoked(
        diff, axes=diff_ax, show=False, plot_type="butterfly", exclude=[], **ts_args_def
    )
    # handle title
    # we use a new axis for the title to handle scaling of plots
    old_title = ts_ax.get_title()
    ts_ax.set_title("")

    if title is not None:
        if title == "":
            title = old_title
        fig.suptitle(title)

    # topomap
    contours = topomap_args.get("contours", 6)
    ch_type = ch_types.pop()  # set should only contain one element
    # Since the data has all the ch_types, we get the limits from the plot.
    vmin, vmax = ts_ax.get_ylim()
    norm = ch_type == "grad"
    vmin = 0 if norm else vmin
    vmin, vmax = _setup_vmin_vmax(evoked_cleaned.data, vmin, vmax, norm)
    if not isinstance(contours, list | np.ndarray):
        locator, contours = _set_contour_locator(vmin, vmax, contours)
    else:
        locator = None

    topomap_args_pass = dict(extrapolate="local") if ch_type == "seeg" else dict()
    topomap_args_pass.update(topomap_args)
    topomap_args_pass["outlines"] = topomap_args.get("outlines", "head")
    topomap_args_pass["contours"] = contours
    evoked_cleaned.plot_topomap(
        times=times_sec, axes=map_ax, show=False, colorbar=False, **topomap_args_pass
    )

    if topomap_args.get("colorbar", True):
        from matplotlib import ticker

        cbar = fig.colorbar(map_ax[0].images[0], ax=map_ax, cax=cbar_ax, shrink=0.8)
        cbar.ax.grid(False)  # auto-removal deprecated as of 2021/10/05
        if isinstance(contours, list | np.ndarray):
            cbar.set_ticks(contours)
        else:
            if locator is None:
                locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = locator
        cbar.update_ticks()

    # connection lines
    # draw the connection lines between time series and topoplots
    for timepoint, map_ax_ in zip(times_ts, map_ax):
        con = ConnectionPatch(
            xyA=[timepoint, ts_ax.get_ylim()[1]],
            xyB=[0.5, 0],
            coordsA="data",
            coordsB="axes fraction",
            axesA=ts_ax,
            axesB=map_ax_,
            color="grey",
            linestyle="-",
            linewidth=1.5,
            alpha=0.66,
            zorder=1,
            clip_on=False,
        )
        fig.add_artist(con)

    # mark times in time series plot
    for timepoint in times_ts:
        ts_ax.axvline(
            timepoint, color="grey", linestyle="-", linewidth=1.5, alpha=0.66, zorder=0
        )

    # show and return it
    plt_show(show)
    return fig


def plot_waveform_masr(epochs, cleaned_avg_picard, cleaned_avg_ssp,
                       cleaned_avg_asr, cleaned_avg_masr, cleaned_avg_opt, cleaned_avg_xdawn, channel_names):
    # 定义比较顺序和对应标题（顺序为：Picard-ICA, SSP, ASR, MASR, OTP, Xdawn）
    cleaned_avgs = [cleaned_avg_picard, cleaned_avg_ssp, cleaned_avg_asr, cleaned_avg_masr, cleaned_avg_opt,
                    cleaned_avg_xdawn]
    titles = ["MASR", "ASR", "Picard-ICA", "SSP", "OTP", "XDAWN"]

    # 遍历每个处理方法，生成联合图
    for evoked, title in zip(cleaned_avgs, titles):
        plot_evoked_joint(
            evoked,
            epochs,
            times="peaks",
            title=None,
            picks=channel_names,
            exclude=None,
            show=True,
        )
