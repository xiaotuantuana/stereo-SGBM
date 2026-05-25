# 纯视觉双目深度相机软件

这是一个基于 Python、OpenCV 和 Tkinter 的双目视频 / 双目相机测距软件。

## 直接运行

Windows 下双击：

```text
启动双目深度相机软件.bat
```

首次运行会自动创建 `.venv` 虚拟环境并安装依赖，需要联网。安装完成后会启动软件界面。

也可以手动运行：

```bash
pip install -r requirements.txt
python stereo_depth_camera_app.py
```

## 仓库内容

- `stereo_depth_camera_app.py`：软件主程序。
- `启动双目深度相机软件.bat`：Windows 一键启动脚本。
- `requirements.txt`：运行依赖。
- `双目深度相机软件说明.md`：详细使用说明。

## 说明

软件支持左右拼接 AVI 视频，也支持识别系统相机接口。只识别到 1 个相机时只显示左图；识别到 2 个相机时使用前两个接口作为左右相机进行双目测距。

下载后可以直接启动软件；如果仓库中没有示例视频，请在界面左上角点击“选择视频”选择自己的左右拼接 AVI 文件，或点击“识别相机”使用相机输入。
