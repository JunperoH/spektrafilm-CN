"""Simplified Chinese presentation text for the spektrafilm desktop GUI.

Only text shown to the user is translated. Runtime parameter names, enum values,
profile identifiers, and saved GUI state remain in their original form.
"""

from __future__ import annotations


UI_TEXT = {
    # Navigation and panels.
    'main': '主界面',
    'film': '胶片',
    'print': '相纸',
    'advanced': '高级',
    'config': '设置',
    'import rgb': '导入 RGB 图像',
    'import raw': '导入 RAW 图像',
    'crop and upscale': '裁剪与放大',
    'input': '输入',
    'camera': '相机',
    'profiles': '胶片与相纸',
    'exposure control': '曝光控制',
    'enlarger': '放大机',
    'scanner': '扫描仪',
    'output': '输出',
    'halation': '光晕',
    'couplers': 'DIR 耦合剂',
    'grain': '颗粒',
    'camera diffusion': '相机柔焦滤镜',
    'enlarger diffusion': '放大机柔焦滤镜',
    'diffusion': '柔焦滤镜',
    'chemistry': '显影化学',
    'glare': '眩光',
    'preflash': '预闪光',
    'spectral upsampling': '光谱重建',
    'input gamut compress': '输入色域压缩',
    'output gamut compress': '输出色域压缩',
    'tune': '微调',
    'experimental': '实验功能',
    'gui parameters': '界面参数',
    'display': '显示',
    'napari layers': '图层',
    # Actions and messages.
    'select file': '选择文件',
    'select input raw': '选择 RAW 文件',
    'select input image': '选择图像',
    'no raw selected': '尚未选择 RAW 文件',
    'no image selected': '尚未选择图像',
    'reprocess raw': '重新处理 RAW',
    'save current as default': '设为默认参数',
    'save current to file': '保存参数到文件',
    'load from file': '从文件加载参数',
    'restore factory default': '恢复出厂参数',
    'preview': '预览',
    'scan': '完整渲染',
    'save': '保存',
    'update': '更新',
    'ccw rotate': '逆时针旋转',
    'cw rotate': '顺时针旋转',
    'reset view': '重置视图',
    'ready': '就绪',
    'scan for print': '按相纸扫描',
    # Common parameter names.
    'active': '启用',
    'algorithm': '算法',
    'amount': '强度',
    'auto exposure': '自动曝光',
    'auto exposure method': '测光方式',
    'auto preview': '自动预览',
    'black correction': '黑点校正',
    'black level': '黑电平',
    'bloom intensity': '泛光强度',
    'bloom size': '泛光范围',
    'blur': '模糊',
    'blur dye clouds um': '染料云模糊（μm）',
    'boost ev': '高光增强（EV）',
    'boost range': '高光增强范围',
    'core intensity': '核心强度',
    'core size': '核心范围',
    'crop': '裁剪',
    'crop center': '裁剪中心',
    'crop size': '裁剪尺寸',
    'density min': '最低密度',
    'developer exhaustion': '显影液耗竭',
    'diffusion size um': '扩散范围（μm）',
    'exposure compensation ev': '曝光补偿（EV）',
    'film channel swap': '胶片通道交换',
    'film format mm': '胶片画幅（mm）',
    'film gamma factor': '胶片伽马系数',
    'film profile': '胶片型号',
    'filter family': '滤镜类型',
    'gamma factor': '伽马系数',
    'gamma factor blue': '蓝色伽马系数',
    'gamma factor fast': '快层伽马系数',
    'gamma factor green': '绿色伽马系数',
    'gamma factor red': '红色伽马系数',
    'gamma factor slow': '慢层伽马系数',
    'gamma interlayer b to rg': '蓝层→红/绿层抑制',
    'gamma interlayer g to rb': '绿层→红/蓝层抑制',
    'gamma interlayer r to gb': '红层→绿/蓝层抑制',
    'gamma samelayer rgb': '各通道同层抑制',
    'gray 18% canvas': '18% 灰色画布',
    'halation amount': '光晕强度',
    'halation bounce decay': '反射光晕衰减',
    'halation first sigma um': '首次反射范围（μm）',
    'halation n bounces': '光晕反射次数',
    'halation renormalize': '光晕能量归一化',
    'halation spatial scale': '光晕范围倍率',
    'halation strength': '光晕强度',
    'halo intensity': '晕光强度',
    'halo size': '晕光范围',
    'halo warmth': '晕光暖调',
    'inhibition interlayer': '层间抑制',
    'inhibition samelayer': '同层抑制',
    'apply cctf decoding': '应用 CCTF 解码',
    'input color space': '输入色彩空间',
    'knee': '拐点',
    'lens blur': '镜头模糊',
    'lens blur um': '镜头模糊（μm）',
    'micro structure': '微观结构',
    'output color space': '输出色彩空间',
    'output interpolation': '输出插值方式',
    'particle area um2': '颗粒面积（μm²）',
    'particle scale': '颗粒比例',
    'particle scale layers': '各层颗粒比例',
    'percent': '百分比',
    'exposure': '曝光量',
    'm filter shift': '洋红滤镜偏移',
    'y filter shift': '黄色滤镜偏移',
    'preview max size': '预览最大尺寸',
    'print channel swap': '相纸通道交换',
    'print exposure': '相纸曝光',
    'print auto compensation': '相纸自动补偿',
    'print illuminant': '放大机光源',
    'print m filter shift': '洋红滤镜偏移',
    'print profile': '相纸型号',
    'print y filter shift': '黄色滤镜偏移',
    'protect ev': '高光保护（EV）',
    'roughness': '粗糙度',
    'saving cctf encoding': '保存时应用 CCTF 编码',
    'saving color space': '保存色彩空间',
    'scan film': '扫描胶片',
    'simulation': '模拟',
    'scatter amount': '散射强度',
    'scatter core um': '散射核心范围（μm）',
    'scatter spatial scale': '散射范围倍率',
    'scatter tail um': '散射拖尾范围（μm）',
    'scatter tail weight': '散射拖尾权重',
    'spatial scale': '空间范围倍率',
    'spectral gaussian blur': '光谱高斯模糊',
    'strength': '强度',
    'sublayers active': '启用子层',
    'uniformity': '均匀度',
    'unsharp mask': '锐化蒙版',
    'upscale factor': '放大倍率',
    'use display transform': '使用显示器色彩转换',
    'white correction': '白点校正',
    'white level': '白电平',
    'white padding': '白边宽度',
    'hanatos2025 adaptation surface': 'Hanatos2025 表面适应',
    'hanatos2025 adaptation window': 'Hanatos2025 窗口适应',
    # RAW controls and option labels.
    'white balance': '白平衡',
    'temperature': '色温',
    'tint': '色调',
    'lens correction': '镜头校正',
    'as_shot': '拍摄时设置',
    'daylight': '日光',
    'tungsten': '钨丝灯',
    'custom': '自定义',
    'center_weighted': '中央重点测光',
    'matrix': '矩阵测光',
    'multi_zone': '多区域测光',
    'partial': '局部测光',
    'highlight_weighted': '高光重点测光',
    'median': '中位数测光',
    'average': '平均测光',
    'nearest': '最近邻',
    'linear': '线性',
    'cubic': '三次插值',
    'spline16': '样条 16',
    'spline36': '样条 36',
    'lanczos': 'Lanczos 插值',
    'blackman': 'Blackman 插值',
    'off': '关闭',
    'cine': '电影',
    'still': '摄影',
}


def translate_text(text: str) -> str:
    """Translate a static GUI label, preserving unknown technical names."""
    return UI_TEXT.get(text.casefold(), text.lower())


def translate_option(value: str) -> str:
    """Translate an enum's display text without changing its stored value."""
    return UI_TEXT.get(value.casefold(), value)


DIALOG_TEXT = {
    'Load raw': '导入 RAW 图像',
    'Save output': '保存输出',
    'Save output image': '保存输出图像',
    'Run simulation': '运行模拟',
    'Save current as default': '设为默认参数',
    'Save GUI state': '保存界面参数',
    'Load GUI state': '加载界面参数',
    'Restore factory default': '恢复出厂参数',
    'Run a simulation before saving the output layer.': '请先运行模拟，再保存输出图像。',
    'Load an input image before running the simulation.': '请先导入图像，再运行模拟。',
    'Failed to load RAW image.': '导入 RAW 图像失败。',
    'Failed to save output image.': '保存输出图像失败。',
    'Simulation failed.': '模拟失败。',
    'Failed to save default GUI state.': '保存默认界面参数失败。',
    'Failed to save GUI state.': '保存界面参数失败。',
    'Failed to load GUI state.': '加载界面参数失败。',
    'Failed to clear the saved startup default.': '清除启动默认参数失败。',
}


def translate_dialog(text: str) -> str:
    """Translate a dialog title or message while retaining exception details."""
    head, separator, detail = text.partition('\n\n')
    translated = DIALOG_TEXT.get(head, head)
    return translated + separator + detail


STATUS_TEXT = {
    'Loading raw...': '正在导入 RAW 图像…',
    'Load raw failed': '导入 RAW 图像失败',
    'Loaded raw, lens correction not applied': 'RAW 图像已导入，未应用镜头校正',
    'Loaded raw': 'RAW 图像已导入',
    'Display transform unavailable: no display profile detected, disabled': '显示器色彩转换不可用：未检测到显示器配置文件，已关闭',
    'Display transform: disabled': '显示器色彩转换：已关闭',
    'Display transform: no display profile, using raw preview': '显示器色彩转换：无显示器配置文件，使用原始预览',
    'Display transform: transform failed, using raw preview': '显示器色彩转换失败，使用原始预览',
    'Simulation already running': '模拟正在运行',
    'Saved current GUI state as the startup default': '当前界面参数已设为启动默认值',
    'Restored factory default GUI state': '已恢复出厂界面参数',
}


def translate_status(message: str) -> str:
    """Translate status-bar messages without changing paths or error details."""
    if message in STATUS_TEXT:
        return STATUS_TEXT[message]
    prefixes = {
        'Loaded raw and applied lens correction: ': 'RAW 图像已导入，并应用镜头校正：',
        'Saved output image to ': '输出图像已保存至 ',
        'Saved GUI state to ': '界面参数已保存至 ',
        'Loaded GUI state from ': '界面参数已从此文件加载：',
        'Display transform: display profile found (': '显示器色彩转换：找到显示器配置文件（',
        'Display transform: active (': '显示器色彩转换：已启用（',
    }
    for english, chinese in prefixes.items():
        if message.startswith(english):
            translated = chinese + message[len(english):]
            if english.startswith('Display transform:'):
                return translated.replace(')', '）')
            return translated.replace(', but failed to copy metadata: ', '，但复制元数据失败：')
    if message.startswith('Computing ') and message.endswith('...'):
        mode = message[len('Computing '):-3].casefold()
        return f'正在计算{UI_TEXT.get(mode, mode)}…'
    for mode in ('Preview', 'Scan', 'Simulation'):
        if message == f'{mode} failed':
            return f'{UI_TEXT.get(mode.casefold(), mode)}失败'
        prefix = f'{mode} completed. '
        if message.startswith(prefix):
            return f'{UI_TEXT.get(mode.casefold(), mode)}完成。{translate_status(message[len(prefix):])}'
    return message
