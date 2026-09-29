# Spektrafilm GPU 中文版

基于 [Andrea Volpato 的 spektrafilm](https://github.com/andreavolpato/spektrafilm) 和 [Célian Le Bolloch 的 Windows GPU 社区版](https://github.com/natcelian/spektrafilm.exe) 1.0.2 整理。此仓库提供中文 GUI 文案、FILM/PRINT 参数提示、中文教程，以及 Windows 可携带版构建文件。

## 下载与运行

在 [Releases](https://github.com/JunperoH/spektrafilm-CN/releases) 下载 `Spektrafilm-GPU-zh-CN-Windows-x64.zip`，解压后运行 `Spektrafilm/Spektrafilm.exe`。不要只取出 exe：旁边的 `_internal` 文件夹也是程序的一部分。需要 Vulkan 驱动与支持 Vulkan 的显卡才能启用 GPU；在界面底部可以关闭 GPU 加速，改用 CPU。

这份 Windows 构建在 Windows 11、Python 3.13.14、AMD Radeon 780M 上完成启动与 Vulkan GPU 初始化验证。其他显卡和 Windows 版本请以实际运行结果为准。

## 中文说明

- [使用说明](docs/GPU中文版-使用说明.md)
- [参数入门](docs/Spektrafilm-参数入门.md)
- [FILM 与 PRINT 每项参数详解](docs/Spektrafilm-FILM-PRINT-参数详解.md)

六个主页面和 FILM/PRINT 参数提示已中文化。胶片与相纸配置文件的内部键名、部分型号名、第三方组件文字仍可能显示英文；中文化只改变界面文字，不改变模拟算法与预设格式。

## 从源码运行

要求 Windows x64 和 Python 3.13：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m spektrafilm_gui.app
```

构建可携带版：

```powershell
.\build-windows.ps1
```

输出位于 `dist/Spektrafilm/`。构建脚本和 `requirements-build.txt` 记录了本次 Windows 构建使用的 Python 依赖版本。

## 来源与许可

核心软件版权属于 Andrea Volpato；社区 GPU 与 Windows 打包修改来自 Célian Le Bolloch。本仓库在这些工作上增加中文界面、参数说明和可重复构建配置。代码遵循 [GPL-3.0-or-later](LICENSE)，原有社区声明见 [NOTICE](NOTICE)。胶片配置文件、LUT 及其直接衍生物适用单独的 [CC BY-SA 4.0 许可说明](SPEKTRAFILM_LICENSE.txt)。第三方依赖各自保留原许可证。

本仓库不是上述两个上游项目的官方发行版。问题反馈请先提供 Windows 版本、显卡、GPU/CPU 开关状态和复现步骤。

