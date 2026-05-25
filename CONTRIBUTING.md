# 贡献说明

欢迎提交 Issue 或 Pull Request 来改进软件。

## 提交问题时请提供

- 操作系统版本。
- Python 版本。
- 输入类型：视频 / 单相机 / 双相机。
- 视频分辨率或相机采集分辨率。
- 是否使用手动标定参数。
- 报错截图或终端输出。

## 开发检查

提交前至少运行：

```bash
python -m py_compile stereo_depth_camera_app.py
```

如果改动了依赖，请同步更新 `requirements.txt` 和 README。
