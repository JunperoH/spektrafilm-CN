"""Chinese presentation layer for the community GPU build.

All serialized values and profile identifiers remain in English.
"""
from __future__ import annotations

from dataclasses import replace

from spektrafilm_gui.localization_zh_cn import UI_TEXT
from spektrafilm_gui.tooltips_zh_cn import PARAM_TOOLTIPS, DIFFUSION_TOOLTIPS, translate_button_tooltip


EXTRA_TEXT = {
    'scan': '扫描', 'scan for print': '相纸扫描', 'preview': '预览', 'save': '保存',
    'age (years)': '存放年数', 'aperture sigma um': '孔径宽度 μm',
    'black point': '黑点', 'black point r/g/b': '红/绿/蓝黑点',
    'camera auto exposure': '相机自动曝光', 'camera compensation ev': '相机曝光补偿 EV',
    'camera lens blur um': '相机镜头模糊 μm', 'cell size ratio': '网格尺寸比例',
    'color adaptation': '色彩适配', 'color separation': '颜色分离',
    'contrast': '对比度', 'contrast trim': '对比度微调',
    'coverage epsilon': '覆盖阈值', 'crossover r/g/b': '红/绿/蓝交叉影响',
    'cyan / red': '青 / 红', 'desqueeze': '反变形宽银幕',
    'development time min': '显影时间（分钟）', 'diffusion active': '启用柔焦',
    'diffusion family': '柔焦滤镜类型', 'diffusion strength': '柔焦强度',
    'dye density': '染料密度', 'encode tones (normal)': '编码色调（标准）',
    'export format': '导出格式', 'fast-layer gamma': '快感光层伽马',
    'fog r/g/b': '红/绿/蓝灰雾', 'full-precision re-render': '全精度重新渲染',
    'gamma': '伽马', 'grain gain': '颗粒增益', 'grain model': '颗粒模型',
    'highlight shape': '高光曲线形状', 'highlight tint hue': '高光色调色相',
    'highlight tint strength': '高光色调强度', 'hue sat c/b/m': '青/蓝/洋红饱和度',
    'hue sat r/y/g': '红/黄/绿饱和度', 'ir filter': '红外滤镜',
    'layer scale': '分层尺寸倍率', 'layered': '分层模拟',
    'leuco-cyan coupling': '无色青染料耦合', 'magenta / green': '洋红 / 绿',
    'max grains per cell': '每格最大颗粒数', 'max radius quantile': '最大半径分位数',
    'mean radius um': '平均半径 μm', 'micro contrast': '微对比度',
    'mode': '模式', 'negative bleach bypass': '负片跳漂白',
    'print c filter shift': '相纸青色滤镜偏移', 'print bleach bypass': '相纸跳漂白',
    'print gamma factor': '相纸伽马倍率', 'push / pull stops': '迫冲 / 减冲档数',
    'rms granularity': 'RMS 颗粒度', 'rms strength': 'RMS 强度',
    'radius scale rgb': '红/绿/蓝半径倍率', 'radius stddev ratio': '半径标准差比例',
    'samples': '采样数', 'saturation': '饱和度',
    'scan black correction': '扫描黑点校正', 'scan black level': '扫描黑电平',
    'scan exposure': '扫描曝光', 'scan lens blur': '扫描镜头模糊',
    'scan unsharp mask': '扫描锐化蒙版', 'scan white correction': '扫描白点校正',
    'scan white level': '扫描白电平', 'shadow shape': '暗部曲线形状',
    'shadow tint hue': '暗部色调色相', 'shadow tint strength': '暗部色调强度',
    'shoulder': '肩部', 'skew': '倾斜角度', 'slow-layer gamma': '慢感光层伽马',
    'speed loss ev': '感光度损失 EV', 'storage': '保存环境',
    'synthesis amount': '合成颗粒量', 'synthesis quality': '合成质量',
    'synthesis sharpness': '合成锐度', 'synthesis size': '合成颗粒尺寸',
    'taking filter': '拍摄滤镜', 'toe': '趾部', 'uv filter': '紫外滤镜',
    'vibrance': '自然饱和度', 'white point': '白点',
    'white point r/g/b': '红/绿/蓝白点', 'yellow / blue': '黄 / 蓝',
    'gpu acceleration (vulkan)': 'GPU 加速（Vulkan）',
    'c filter shift': '青色滤镜偏移',
    'input': '输入', 'edit': '编辑', 'roll': '胶卷',
    'cancel analysis': '取消分析', 'analyze roll': '分析整卷',
    'cancel export': '取消导出', 'export': '导出',
    'metadata': '元数据', 'rating': '评分', 'linear tones': '线性色调',
    'sync export settings': '同步导出设置', 'sync metadata': '同步元数据',
    'add files to roll': '向胶卷添加文件',
    'add files': '添加文件', 'add texture': '添加纹理',
    'apply look to all': '将效果应用到全部', 'artist': '作者',
    'auto levels': '自动色阶', 'camera model': '相机型号',
    'clear': '清除', 'copyright': '版权', 'delete': '删除',
    'description': '说明', 'export lut...': '导出 LUT…',
    'export roll': '导出整卷',
    'exports every frame in the roll.': '导出胶卷中的每一帧。',
    'film curves': '胶片曲线', 'film format': '胶片画幅',
    'film stocks folder': '胶片配置文件夹', 'filter shifts': '滤镜偏移',
    'gains ev': '增益 EV', 'gamma rgb': '红/绿/蓝伽马',
    'histogram': '直方图', 'lens model': '镜头型号',
    'logs folder': '日志文件夹', 'open logs': '打开日志',
    'open roll': '打开胶卷', 'padding burn': '白边曝光',
    'presets folder': '预设文件夹', 'remove': '移除',
    'reset': '重置', 'save preset...': '保存预设…',
    'save roll': '保存胶卷', 'straighten': '拉直',
    'swap': '交换', 'use roll exposure': '沿用整卷曝光',
    'full screen': '全屏', 'grain preview': '颗粒预览',
    'mirror h': '水平镜像', 'mirror v': '垂直镜像',
    'reprocess rgb': '重新处理 RGB',
    'cpu fallback (gpu off)': '已切换至 CPU（GPU 已关闭）',
    'gpu status unavailable': '无法获取 GPU 状态',
    'cpu only (no gpu detected)': '仅使用 CPU（未检测到 GPU）',
}

EXTRA_TOOLTIPS = {
    ('simulation', 'camera_lens_blur_um'): '相机镜头模糊的高斯标准差，单位 μm；普通镜头约 5 μm，优质镜头约 2–4 μm。',
    ('simulation', 'camera_diffusion_filter_family'): '选择拍摄镜头前的柔焦滤镜类型。',
    ('simulation', 'print_illuminant'): '选择虚拟放大机的光源类型。',
    ('simulation', 'print_y_filter_shift'): '彩色放大机黄色滤镜相对中性位置的偏移。',
    ('simulation', 'print_m_filter_shift'): '彩色放大机洋红滤镜相对中性位置的偏移。',
    ('simulation', 'print_c_filter_shift'): '彩色放大机青色滤镜相对中性位置的偏移。',
    ('simulation', 'print_shadow_shape'): '调整相纸暗部密度曲线的形状。',
    ('simulation', 'print_highlight_shape'): '调整相纸高光密度曲线的形状。',
    ('simulation', 'negative_bleach_bypass'): '跳过负片的漂白步骤，使银颗粒保留，通常增加对比度并降低饱和度。',
    ('simulation', 'negative_leuco_cyan_coupling'): '控制负片无色青染料的耦合效果。',
    ('simulation', 'print_bleach_bypass'): '跳过相纸的漂白步骤，使银颗粒保留。',
    ('simulation', 'film_development_time'): '胶片显影时间，单位分钟；改变负片的密度曲线。',
    ('simulation', 'diffusion_filter_family'): '选择放大机阶段的柔焦滤镜类型。',
    ('simulation', 'scan_lens_blur'): '扫描仪镜头模糊的高斯标准差，单位像素。',
    ('simulation', 'scan_white_correction'): '启用扫描结果的白点校正。',
    ('simulation', 'scan_white_level'): '白点校正所用的目标白电平。',
    ('simulation', 'scan_black_correction'): '启用扫描结果的黑点校正。',
    ('simulation', 'scan_black_level'): '黑点校正所用的目标黑电平。',
    ('simulation', 'scan_unsharp_mask'): '扫描结果的锐化蒙版参数。',
    ('simulation', 'export_format'): '选择导出图像的文件格式。',
    ('simulation', 'export_full_precision'): '导出时以完整精度重新渲染，可能比预览慢。',
    ('display', 'use_display_transform'): '按显示器 ICC 配置文件转换预览颜色。',
    ('display', 'output_interpolation'): '选择预览窗口放大或缩小时的插值方式。',
    ('display', 'white_padding'): '在预览图像周围增加白边；数值为相对图像长边的比例。',
    ('display', 'preview_max_size'): '预览图像长边的最大像素数；降低可加快预览。',
    ('special', 'film_gamma_factor'): '负片密度曲线的伽马倍率；小于 1 降低对比度，大于 1 提高对比度。',
    ('special', 'print_gamma_factor'): '相纸密度曲线的伽马倍率。',
    ('input_gamut_compress', 'active'): '在光谱重建前，将输入色度压缩到可见光谱轨迹附近。',
    ('input_gamut_compress', 'algorithm'): '输入色域压缩算法；xy 在色度图中处理，Oklch 在感知空间中处理。',
    ('input_gamut_compress', 'knee'): '输入色度压缩曲线的阈值、上限和幂次。',
    ('output_gamut_compress', 'algorithm'): '输出色域压缩算法；影响超出目标色域的颜色如何映射。',
    ('output_gamut_compress', 'knee'): '输出色度压缩曲线的阈值、上限和幂次。',
    ('scan_finishing', 'crop'): '启用扫描图像裁剪。',
    ('scan_finishing', 'crop_center'): '裁剪中心在原图中的相对坐标。',
    ('scan_finishing', 'crop_size'): '裁剪范围相对于原图长边的尺寸。',
    ('scan_finishing', 'skew_deg'): '校正扫描图像的倾斜角度，单位度。',
    ('scan_finishing', 'desqueeze'): '还原变形宽银幕镜头压缩过的画面宽度。',
    ('scan_finishing', 'active'): '启用扫描结果的后期调整。',
    ('scan_finishing', 'exposure_ev'): '扫描结果的曝光调整，单位 EV。',
    ('scan_finishing', 'gain_red_ev'): '扫描结果红色通道的曝光增益，单位 EV。',
    ('scan_finishing', 'gain_green_ev'): '扫描结果绿色通道的曝光增益，单位 EV。',
    ('scan_finishing', 'gain_blue_ev'): '扫描结果蓝色通道的曝光增益，单位 EV。',
    ('scan_finishing', 'color_separation'): '调整扫描结果不同颜色之间的分离程度。',
    ('scan_finishing', 'black_point'): '设置整体黑点；提高可压暗最暗区域。',
    ('scan_finishing', 'white_point'): '设置整体白点；降低可提亮最亮区域。',
    ('scan_finishing', 'black_point_rgb'): '分别设置红、绿、蓝通道的黑点。',
    ('scan_finishing', 'white_point_rgb'): '分别设置红、绿、蓝通道的白点。',
    ('scan_finishing', 'gamma'): '调整中间调亮度；改变亮度曲线的伽马。',
    ('scan_finishing', 'contrast'): '调整扫描结果的整体对比度。',
    ('scan_finishing', 'toe'): '调整暗部曲线的趾部形状。',
    ('scan_finishing', 'shoulder'): '调整高光曲线的肩部形状。',
    ('scan_finishing', 'micro_contrast'): '调整局部细节的微对比度。',
    ('scan_finishing', 'saturation'): '调整整体颜色饱和度。',
    ('scan_finishing', 'vibrance'): '优先增强较低饱和度颜色的自然饱和度。',
    ('scan_finishing', 'hue_saturation_ryg'): '分别调整红、黄、绿三个色相区域的饱和度。',
    ('scan_finishing', 'hue_saturation_cbm'): '分别调整青、蓝、洋红三个色相区域的饱和度。',
    ('scan_finishing', 'shadow_tint_hue'): '设置暗部着色的色相。',
    ('scan_finishing', 'shadow_tint_strength'): '设置暗部着色的强度。',
    ('scan_finishing', 'highlight_tint_hue'): '设置高光着色的色相。',
    ('scan_finishing', 'highlight_tint_strength'): '设置高光着色的强度。',
    ('input_image', 'crop'): '按原图比例裁剪，用于预览细节。',
    ('input_image', 'crop_center'): '输入图像裁剪中心的相对坐标。',
    ('input_image', 'crop_size'): '输入图像裁剪区域的相对尺寸。',
    ('input_image', 'input_color_space'): '输入图像的色彩空间；内部会转换为工作色彩空间。',
    ('input_image', 'apply_cctf_decoding'): '对输入图像应用逆传递函数，转换为线性光。',
    ('input_image', 'upscale_factor'): '输入图像尺寸的放大倍率。',
    ('input_image', 'spectral_upsampling_method'): '选择把 RGB 图像重建为光谱的算法。',
    ('input_image', 'apply_hanatos2025_adaptation_window'): '重建光谱时应用 Hanatos2025 带通适应窗口。',
    ('input_image', 'apply_hanatos2025_adaptation_surface'): '重建光谱时应用 Hanatos2025 表面适应多项式。',
    ('input_image', 'spectral_gaussian_blur'): '重建光谱上的高斯模糊标准差，单位 nm。',
    ('input_image', 'filter_uv'): '控制输入光谱的紫外线过滤。',
    ('input_image', 'filter_ir'): '控制输入光谱的红外线过滤。',
    ('input_image', 'color_filter'): '选择拍摄时使用的色彩滤镜。',
    ('load_raw', 'white_balance'): '选择 RAW 图像导入时的白平衡模式。',
    ('load_raw', 'temperature'): '自定义白平衡色温，单位 K。',
    ('load_raw', 'tint'): '自定义白平衡的绿色/洋红色调。',
    ('load_raw', 'lens_correction'): '导入 RAW 时应用镜头校正。',
    ('load_rgb', 'white_balance'): '选择 RGB 图像导入时的白平衡模式。',
    ('load_rgb', 'temperature'): 'RGB 自定义白平衡的色温，单位 K。',
    ('load_rgb', 'tint'): 'RGB 自定义白平衡的色调。',
    ('load_rgb', 'recover_16bit'): '对显示编码的图像增加少量对比度和色彩密度，以接近 RAW 导入结果。',
    ('grain', 'model'): '选择颗粒算法：传统颗粒、RMS 颗粒度或逐粒合成。算法不同，下方生效的参数也不同。',
    ('grain', 'rms_granularity'): '胶片技术资料中的 RMS 颗粒度数值；决定统计颗粒噪声的基准幅度。',
    ('grain', 'rms_strength'): 'RMS 颗粒模型的效果倍率；调大后颗粒更明显。',
    ('grain', 'synthesis_size'): '逐粒合成颗粒的整体尺寸倍率。',
    ('grain', 'synthesis_amount'): '逐粒合成颗粒的覆盖量或可见强度。',
    ('grain', 'synthesis_sharpness'): '逐粒合成颗粒边缘的清晰度。',
    ('grain', 'synthesis_quality'): '逐粒合成的质量档位；较高质量通常增加计算量。',
    ('grain', 'synthesis_samples'): '合成颗粒的采样数量；增加可改善统计稳定性，也会延长计算时间。',
    ('grain', 'synthesis_mean_radius_um'): '合成颗粒的平均半径，单位 μm。',
    ('grain', 'synthesis_radius_stddev_ratio'): '颗粒半径标准差与平均半径之比；越大，颗粒大小差异越明显。',
    ('grain', 'synthesis_aperture_sigma_um'): '成像孔径的模糊范围，单位 μm；影响颗粒边缘。',
    ('grain', 'synthesis_cell_size_ratio'): '空间网格相对颗粒尺寸的比例；主要影响合成效率。',
    ('grain', 'synthesis_max_radius_quantile'): '限制极端大颗粒的半径分位数。',
    ('grain', 'synthesis_coverage_epsilon'): '颗粒覆盖计算的容差；数值越小，计算越精细。',
    ('grain', 'synthesis_max_grains_per_cell'): '每个空间网格容纳的最大颗粒数；过低可能截断高密度区域。',
    ('grain', 'synthesis_radius_scale'): '红、绿、蓝乳剂层的合成颗粒半径倍率。',
    ('grain', 'synthesis_layered'): '分别模拟各乳剂层的颗粒结构。',
    ('grain', 'synthesis_layer_scale'): '各乳剂层颗粒尺寸的倍率。',
    ('push_pull', 'active'): '启用迫冲或减冲模拟；会改变胶片显影曲线。',
    ('push_pull', 'mode'): '选择显影调整方式。',
    ('push_pull', 'stops'): '迫冲或减冲的档数（EV）；正值迫冲，负值减冲。',
    ('push_pull', 'gamma_factor_fast'): '迫冲/减冲时快感光层的伽马倍率。',
    ('push_pull', 'gamma_factor_slow'): '迫冲/减冲时慢感光层的伽马倍率。',
    ('push_pull', 'developer_exhaustion'): '模拟显影液耗竭对高密度区域曲线的影响。',
    ('aging', 'active'): '启用胶片老化模拟。',
    ('aging', 'age_years'): '胶片存放年数；与保存条件一起决定老化程度。',
    ('aging', 'storage'): '保存环境；温度和湿度会影响胶片老化速度。',
    ('aging', 'fog_rgb'): '老化产生的红、绿、蓝各层灰雾密度。',
    ('aging', 'gamma'): '老化对整体密度曲线伽马的影响。',
    ('aging', 'gamma_rgb'): '老化对红、绿、蓝各层伽马的影响。',
    ('aging', 'density_scale'): '老化造成的染料密度缩放。',
    ('aging', 'grain_gain'): '老化对颗粒可见程度的增益。',
    ('aging', 'speed_loss_ev'): '老化造成的有效感光度损失，单位 EV。',
    ('preflashing', 'active'): '启用相纸预闪光，在正式曝光前给相纸少量均匀曝光。',
    ('preflashing', 'exposure'): '相纸预闪光的曝光量，单位 EV。',
    ('preflashing', 'y_filter_shift'): '预闪光黄色滤镜相对中性位置的偏移。',
    ('preflashing', 'm_filter_shift'): '预闪光洋红滤镜相对中性位置的偏移。',
    ('preflashing', 'c_filter_shift'): '预闪光青色滤镜相对中性位置的偏移。',
}

_PREFIX = {
    'halation': 'film_render.halation', 'couplers': 'film_render.dir_couplers',
    'grain': 'film_render.grain', 'glare': 'print_render.glare',
    'chemistry': 'print_render.density_curves_morph',
}


def translate_text(text: str) -> str:
    if text.startswith('GPU acceleration: '):
        return 'GPU 加速：' + text[len('GPU acceleration: '):]
    if text.startswith('GPU acceleration ON (Vulkan)'):
        return 'GPU 加速已开启（Vulkan），下次预览或扫描生效'
    if text.startswith('GPU acceleration OFF (CPU fallback)'):
        return 'GPU 加速已关闭，改用 CPU；下次预览或扫描生效'
    if text.startswith('CPU only (no GPU: '):
        return '仅使用 CPU（GPU 错误：' + text[len('CPU only (no GPU: '):]
    return EXTRA_TEXT.get(text.casefold(), UI_TEXT.get(text.casefold(), text))


def translate_spec(section: str, field: str, spec):
    label = translate_text(spec.label) if spec.label else None
    tooltip = EXTRA_TOOLTIPS.get((section, field))
    if tooltip is None:
        prefix = _PREFIX.get(section)
        if prefix:
            tooltip = PARAM_TOOLTIPS.get(f'{prefix}.{field}')
    if tooltip is None and 'diffusion_filter_' in field:
        tooltip = DIFFUSION_TOOLTIPS.get(field.split('diffusion_filter_', 1)[1])
    if tooltip is None and section == 'simulation':
        for prefix in ('camera', 'scanner', 'enlarger', 'io', 'workflow', 'selection'):
            tooltip = PARAM_TOOLTIPS.get(f'{prefix}.{field}')
            if tooltip:
                break
    if tooltip is None:
        tooltip = spec.tooltip
    return replace(spec, label=label, tooltip=tooltip)


def translate_button_spec(spec):
    return replace(spec, text=translate_text(spec.text), tooltip=translate_button_tooltip(spec.tooltip or ''))


def retranslate_widget_tree(root) -> None:
    """Translate static widgets missed by the central spec builders."""
    from qtpy import QtWidgets

    for widget in (root, *root.findChildren(QtWidgets.QWidget)):
        if isinstance(widget, QtWidgets.QTabWidget):
            for index in range(widget.count()):
                title = widget.tabText(index)
                translated = translate_text(title)
                if translated != title:
                    widget.setTabText(index, translated)
        elif isinstance(widget, (QtWidgets.QLabel, QtWidgets.QPushButton,
                                 QtWidgets.QCheckBox, QtWidgets.QGroupBox,
                                 QtWidgets.QRadioButton)):
            original = widget.text() if hasattr(widget, 'text') else widget.title()
            translated = translate_text(original)
            if translated != original:
                if hasattr(widget, 'setText'):
                    widget.setText(translated)
                else:
                    widget.setTitle(translated)
        tooltip = widget.toolTip()
        if tooltip:
            translated = translate_button_tooltip(tooltip)
            if translated != tooltip:
                widget.setToolTip(translated)
