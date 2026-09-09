@echo off
setlocal enabledelayedexpansion
title Climb Analysis Server
pushd "%~dp0"

:: cloudflared writes this itself via --logfile. Shell redirection was tried
:: first and is not worth returning to: the path sits under %TEMP%, which
:: contains a space on any account whose user name does, and every way of
:: quoting that through cmd into a background process failed differently.
set "TUNNEL_LOG=%TEMP%\climb-tunnel.log"
set "LOCAL_URL=http://localhost:8000"

if "%~1"=="" goto menu
if /I "%~1"=="start" goto start
if /I "%~1"=="stop" goto stop
if /I "%~1"=="restart" goto restart
if /I "%~1"=="status" goto status
if /I "%~1"=="local" goto local
goto help

:menu
cls
echo ========================================
echo          Climb Analysis Server
echo ========================================
echo.
echo   [1] Start
echo   [2] Stop
echo   [3] Restart
echo   [4] Status
echo   [5] Exit
echo.
choice /C 12345 /N /M "Choose an option: "
if errorlevel 5 goto done
if errorlevel 4 goto status_menu
if errorlevel 3 goto restart_menu
if errorlevel 2 goto stop_menu
if errorlevel 1 goto start_menu

:start_menu
call :run_start
goto menu_pause

:stop_menu
call :run_stop
goto menu_pause

:restart_menu
call :run_restart
goto menu_pause

:status_menu
call :run_status
goto menu_pause

:menu_pause
echo.
pause
goto menu

:start
call :run_start
goto command_done

:stop
call :run_stop
goto command_done

:restart
call :run_restart
goto command_done

:status
call :run_status
goto command_done

:local
call :run_local
goto command_done

:run_start
echo Starting Climb Analysis and rebuilding changed images...
docker compose up -d --build
if errorlevel 1 goto sub_failed
call :ensure_tunnel
call :show_addresses
exit /b 0

:run_local
echo Starting Climb Analysis without a public link...
docker compose up -d --build
if errorlevel 1 goto sub_failed
echo.
echo   On this machine:  %LOCAL_URL%
exit /b 0

:run_stop
echo Stopping Climb Analysis. Stored data will be kept.
call :stop_tunnel
docker compose stop
if errorlevel 1 goto sub_failed
exit /b 0

:run_restart
echo Restarting Climb Analysis...
docker compose restart
if errorlevel 1 goto sub_failed
call :ensure_tunnel
call :show_addresses
exit /b 0

:run_status
docker compose ps
if errorlevel 1 goto sub_failed
call :show_addresses
exit /b 0

:: ---------------------------------------------------------------- the tunnel
:: A quick tunnel gets a fresh random trycloudflare.com hostname every time it
:: starts and forgets it when it stops, so the address can only be learned by
:: reading what cloudflared logged. Hence the log file and the search below.

:ensure_tunnel
set "TUNNEL_NOTE="
call :find_tunnel_url
if defined PUBLIC_URL exit /b 0

:: Run it rather than just locating it: a name on PATH can be a zero-byte App
:: Execution Alias that resolves and then refuses to launch.
cloudflared --version >nul 2>&1
if errorlevel 1 (
    set "TUNNEL_NOTE=no public link - install cloudflared to get one: winget install --id Cloudflare.cloudflared"
    exit /b 0
)

echo Opening public tunnel...
del "%TUNNEL_LOG%" >nul 2>&1
start "" /b cloudflared tunnel --url %LOCAL_URL% --logfile "%TUNNEL_LOG%"

for /L %%I in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    call :find_tunnel_url
    if defined PUBLIC_URL exit /b 0
)

set "TUNNEL_NOTE=tunnel did not report an address in 30s - see %TUNNEL_LOG%"
exit /b 0

:find_tunnel_url
set "PUBLIC_URL="
:: No process means no tunnel, whatever an old log still says.
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 exit /b 0
if not exist "%TUNNEL_LOG%" exit /b 0
:: Regex over the whole file rather than a pipeline: cloudflared draws the
:: address inside an ASCII box, so the line carries pipe characters, and every
:: pipe in a batch-to-PowerShell command is another thing to escape wrongly.
for /f "usebackq delims=" %%U in (`powershell -NoProfile -Command "[regex]::Match((Get-Content -Raw '%TUNNEL_LOG%' -ErrorAction SilentlyContinue), 'https://[a-z0-9-]+\.trycloudflare\.com').Value"`) do set "PUBLIC_URL=%%U"
exit /b 0

:show_addresses
call :find_tunnel_url
echo.
echo ========================================
echo   On this machine:  %LOCAL_URL%
if defined PUBLIC_URL (
    echo   From anywhere:    !PUBLIC_URL!
    echo ========================================
    echo.
    echo That address is new every time the tunnel starts and dies with it.
    echo It has no login, so anyone with the link can use it. Stop takes it down.
) else (
    echo ========================================
    if defined TUNNEL_NOTE echo   !TUNNEL_NOTE!
)
exit /b 0

:stop_tunnel
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 exit /b 0
echo Closing the public tunnel.
taskkill /IM cloudflared.exe /F >nul 2>&1
del "%TUNNEL_LOG%" >nul 2>&1
exit /b 0

:sub_failed
echo.
echo Command failed. Check that Docker Desktop is running, then try again.
exit /b 1

:help
echo Usage: %~nx0 ^<start^|stop^|restart^|status^|local^>
echo.
echo   start    Start the server, open a public link, and print both addresses
echo   stop     Stop the server and the public link, keeping data
echo   restart  Restart the server and print both addresses
echo   status   Show whether the server is running, and its addresses
echo   local    Start without opening a public link
echo.
echo Double-click the file with no argument to open the interactive menu.
goto command_done

:command_done
set "exit_code=%errorlevel%"
popd
exit /b %exit_code%

:done
popd
exit /b 0
