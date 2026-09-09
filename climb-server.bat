@echo off
setlocal enabledelayedexpansion
title Climb Analysis Server
pushd "%~dp0"

:: cloudflared prints its banner on stderr, but versions differ, so both
:: streams are captured and both are searched. Start-Process refuses to point
:: them at one file, hence two.
set "TUNNEL_LOG=%TEMP%\climb-tunnel.err.log"
set "TUNNEL_OUT=%TEMP%\climb-tunnel.out.log"

if "%~1"=="" goto menu
if /I "%~1"=="start" goto start
if /I "%~1"=="stop" goto stop
if /I "%~1"=="restart" goto restart
if /I "%~1"=="status" goto status
if /I "%~1"=="share" goto share
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
echo   [5] Share (public link)
echo   [6] Exit
echo.
choice /C 123456 /N /M "Choose an option: "
if errorlevel 6 goto done
if errorlevel 5 goto share_menu
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

:share_menu
call :run_share
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

:share
call :run_share
goto command_done

:run_start
echo Starting Climb Analysis and rebuilding changed images...
docker compose up -d --build
if errorlevel 1 goto sub_failed
echo Climb Analysis is available at http://localhost:8000
call :report_tunnel
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
echo Climb Analysis is available at http://localhost:8000
call :report_tunnel
exit /b 0

:run_status
docker compose ps
if errorlevel 1 goto sub_failed
call :report_tunnel
exit /b 0

:: ---------------------------------------------------------------- sharing
:: A quick tunnel gets a fresh random trycloudflare.com hostname every time it
:: starts and forgets it the moment it stops, so the address is only knowable
:: by reading what cloudflared prints. That output goes to a log the launcher
:: then greps, which is why this is not simply echoed.

:run_share
where cloudflared >nul 2>&1
if errorlevel 1 (
    echo cloudflared is not installed. Install it with:
    echo     winget install --id Cloudflare.cloudflared
    exit /b 1
)

call :find_tunnel_url
if defined PUBLIC_URL (
    echo A tunnel is already running.
    call :show_url
    exit /b 0
)

docker compose ps --status running --quiet >nul 2>&1
if errorlevel 1 (
    echo Starting the server first...
    docker compose up -d --build
    if errorlevel 1 goto sub_failed
)

echo.
echo This publishes the app on a public URL. It has no login, so treat the
echo link as "anyone who has it can use it". Stop or close to take it down.
echo.
echo Opening tunnel...
del "%TUNNEL_LOG%" "%TUNNEL_OUT%" >nul 2>&1
:: Launched through Start-Process rather than `cmd /c ... > file`: the log path
:: sits under %TEMP%, which contains a space on any account whose user name
:: does, and the nested quoting that redirect needs misparses there.
powershell -NoProfile -Command "Start-Process -FilePath 'cloudflared' -ArgumentList 'tunnel','--url','http://localhost:8000' -RedirectStandardError '%TUNNEL_LOG%' -RedirectStandardOutput '%TUNNEL_OUT%' -WindowStyle Hidden"
if errorlevel 1 (
    echo Could not launch cloudflared.
    exit /b 1
)

for /L %%I in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    call :find_tunnel_url
    if defined PUBLIC_URL goto share_ready
)

echo Could not read the tunnel address after 30 seconds.
echo Check %TUNNEL_LOG% for what cloudflared reported.
exit /b 1

:share_ready
call :show_url
exit /b 0

:show_url
echo.
echo ========================================
echo   Reachable from anywhere at:
echo   !PUBLIC_URL!
echo ========================================
echo.
echo This address is new every time the tunnel starts and dies with it.
exit /b 0

:find_tunnel_url
:: Cloudflared draws its address inside a box, so the line carries pipe
:: characters that batch cannot echo safely. PowerShell pulls out just the
:: hostname.
set "PUBLIC_URL="
:: No process means no tunnel, whatever an old log still says.
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 exit /b 0
for /f "usebackq delims=" %%U in (`powershell -NoProfile -Command "Get-ChildItem -Path '%TUNNEL_LOG%','%TUNNEL_OUT%' -ErrorAction SilentlyContinue ^| Select-String -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' ^| Select-Object -Last 1 ^| ForEach-Object { $_.Matches.Value }" 2^>nul`) do set "PUBLIC_URL=%%U"
exit /b 0

:report_tunnel
call :find_tunnel_url
if defined PUBLIC_URL call :show_url
exit /b 0

:stop_tunnel
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 exit /b 0
echo Closing the public tunnel.
taskkill /IM cloudflared.exe /F >nul 2>&1
del "%TUNNEL_LOG%" "%TUNNEL_OUT%" >nul 2>&1
exit /b 0

:sub_failed
echo.
echo Command failed. Check that Docker Desktop is running, then try again.
exit /b 1

:help
echo Usage: %~nx0 ^<start^|stop^|restart^|status^|share^>
echo.
echo   start    Build changed images and start the local server
echo   stop     Stop the local server and any public tunnel, keeping data
echo   restart  Restart the existing local server
echo   status   Show whether the local server is running
echo   share    Open a public URL to this machine and print it
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
