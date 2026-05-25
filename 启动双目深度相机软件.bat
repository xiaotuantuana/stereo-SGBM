@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo 正在创建 Python 虚拟环境...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

"%VENV_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo 正在安装运行依赖，首次启动需要联网...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
)

"%VENV_PY%" stereo_depth_camera_app.py
pause
