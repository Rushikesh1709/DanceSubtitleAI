@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo   DanceSubtitleAI - setup and start
echo ============================================================

REM ---- 1. Find Python 3.11 (needed by the pinned MediaPipe version)
set "PYEXE="
py -3.11 --version >nul 2>nul
if not errorlevel 1 set "PYEXE=py -3.11"
if "%PYEXE%"=="" (
  python --version 2>nul | findstr /b /c:"Python 3.11" >nul
  if not errorlevel 1 set "PYEXE=python"
)
if "%PYEXE%"=="" (
  echo [ERROR] Python 3.11 was not found.
  echo         Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  echo         ^(tick "Add python.exe to PATH" and keep the "py launcher" option^), then run this file again.
  pause
  exit /b 1
)
echo [OK] Using: %PYEXE%

REM ---- 2. Create the virtual environment only if it does not exist (never replaced)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 ( echo [ERROR] Could not create the virtual environment. & pause & exit /b 1 )
) else (
  echo [OK] Existing .venv found - keeping it.
)
set "VPY=.venv\Scripts\python.exe"

REM ---- 3. Install dependencies
echo Installing dependencies ^(first time takes a few minutes^) ...
"%VPY%" -m pip install --upgrade pip >nul
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] Dependency installation failed. See README - Troubleshooting. & pause & exit /b 1 )

REM ---- 4. Check OpenCV / MediaPipe versions
"%VPY%" -c "import cv2, mediapipe as mp, numpy; assert hasattr(mp,'solutions'); print('[OK] OpenCV', cv2.__version__, '| MediaPipe', mp.__version__, '| NumPy', numpy.__version__)"
if errorlevel 1 (
  echo [ERROR] MediaPipe/OpenCV check failed. Try:  .venv\Scripts\python.exe -m pip install --force-reinstall mediapipe==0.10.14 numpy==1.26.4
  pause & exit /b 1
)

REM ---- 5. Folders + database
for %%D in (data data\uploads data\processed data\outputs models) do if not exist "%%D" mkdir "%%D"
"%VPY%" init_database.py
if errorlevel 1 ( echo [ERROR] Database initialisation failed. & pause & exit /b 1 )

REM ---- 6. Optional components
echo [INFO] MediaPipe pose/hand/face models are bundled inside the mediapipe package - nothing to download.
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] FFmpeg not found. The app works, but "video with embedded subtitles" and preview copies of MOV/AVI/MKV need FFmpeg.
  echo        See README - FFmpeg. You can also set FFMPEG_PATH in a .env file.
) else (
  echo [OK] FFmpeg found.
)

REM ---- 7. Start
echo.
echo Starting the application. Open  http://127.0.0.1:5000  in your browser.
echo Press CTRL+C in this window to stop.
echo.
"%VPY%" app.py
pause
