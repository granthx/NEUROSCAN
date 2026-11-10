@echo off
REM ===========================================================================
REM Deploy Frontend to Cloudflare Pages
REM ===========================================================================

echo =======================================================
echo   NeuroScan - Cloudflare Pages Deployment Helper
echo =======================================================
echo.

cd frontend

echo [*] Installing dependencies if needed...
call npm install

echo [*] Building Next.js static export...
call npm run build

if %ERRORLEVEL% neq 0 (
    echo [!] Build failed! Please inspect errors above.
    pause
    exit /b 1
)

echo.
echo [*] Build complete! Output located at frontend\out
echo.
echo Would you like to deploy directly using Cloudflare Wrangler CLI? (Y/N)
set /p CHOICE=""

if /i "%CHOICE%"=="Y" (
    echo [*] Running npx wrangler pages deploy out ...
    call npx wrangler pages deploy out --project-name neuroscan-terminal
) else (
    echo.
    echo [*] Git-based Cloudflare deployment instructions:
    echo     1. Push this repository to GitHub.
    echo     2. Go to Cloudflare Dashboard -> Workers & Pages -> Create Application -> Pages.
    echo     3. Connect GitHub repository.
    echo     4. Settings:
    echo        - Root directory: frontend
    echo        - Build command: npm run build
    echo        - Build output directory: out
    echo        - Env var: NEXT_PUBLIC_API_URL = ^<your Hugging Face Space URL^>
)

echo.
pause

