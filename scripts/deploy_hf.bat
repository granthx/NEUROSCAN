@echo off
REM ===========================================================================
REM Deploy Backend to Hugging Face Spaces
REM ===========================================================================

echo =======================================================
echo   NeuroScan - Hugging Face Spaces Deployment Helper
echo =======================================================
echo.

set /p HF_USERNAME="Enter your Hugging Face username: "
set /p SPACE_NAME="Enter your Space name (e.g. neuroscan-backend): "

if "%HF_USERNAME%"=="" (
    echo Error: Username cannot be empty.
    pause
    exit /b 1
)

if "%SPACE_NAME%"=="" (
    echo Error: Space name cannot be empty.
    pause
    exit /b 1
)

set TARGET_DIR=deploy_hf_space
set REMOTE_URL=https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%

echo.
echo [*] Cloning space repo into ./%TARGET_DIR% ...
if exist %TARGET_DIR% (
    echo Target folder %TARGET_DIR% already exists. Cleaning up...
    rmdir /s /q %TARGET_DIR%
)

git clone %REMOTE_URL% %TARGET_DIR%
if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Failed to clone. Make sure you created the Space on huggingface.co first:
    echo     Name: %SPACE_NAME%
    echo     SDK:  Gradio (Blank)
    pause
    exit /b 1
)

echo.
echo [*] Initializing Git LFS...
cd %TARGET_DIR%
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"

echo [*] Copying backend files into space directory...
cd ..
copy Dockerfile %TARGET_DIR%\Dockerfile
copy requirements.txt %TARGET_DIR%\requirements.txt
copy .dockerignore %TARGET_DIR%\.dockerignore
copy .gitattributes %TARGET_DIR%\.gitattributes
copy app.py %TARGET_DIR%\app.py
copy main.py %TARGET_DIR%\main.py

REM Generate Hugging Face YAML Space configuration
(
echo ---
echo title: NeuroScan Brain Tumor Segmentation API
echo emoji: 🧠
echo colorFrom: indigo
echo colorTo: purple
echo sdk: gradio
echo app_file: app.py
echo pinned: false
echo ---
) > %TARGET_DIR%\README.md

echo [*] Copying src code...
xcopy /e /i /y src %TARGET_DIR%\src
echo [*] Copying configs...
xcopy /e /i /y configs %TARGET_DIR%\configs

echo [*] Copying model checkpoints (handled via Git LFS)...
if not exist %TARGET_DIR%\checkpoints mkdir %TARGET_DIR%\checkpoints
copy checkpoints\best_attention_unet.pt %TARGET_DIR%\checkpoints\
if exist checkpoints\best_attention_95.pt copy checkpoints\best_attention_95.pt %TARGET_DIR%\checkpoints\

cd %TARGET_DIR%
git add .
git commit -m "Deploy NeuroScan Brain Tumor Segmentation API"
echo.
echo [*] Pushing to Hugging Face Spaces...
git push origin main

cd ..
rmdir /s /q %TARGET_DIR%

echo.
echo =======================================================
echo  Successfully deployed to Hugging Face!
echo  Your API will be live at:
echo  https://%HF_USERNAME%-%SPACE_NAME%.hf.space
echo =======================================================
pause
