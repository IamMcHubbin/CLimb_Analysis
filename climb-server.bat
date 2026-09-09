@echo off
setlocal
title Climb Analysis Server
pushd "%~dp0"

if "%~1"=="" goto menu
if /I "%~1"=="start" goto start
if /I "%~1"=="stop" goto stop
if /I "%~1"=="restart" goto restart
if /I "%~1"=="status" goto status
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

:run_start
echo Starting Climb Analysis and rebuilding changed images...
docker compose up -d --build
if errorlevel 1 goto sub_failed
echo Climb Analysis is available at http://localhost:8000
exit /b 0

:run_stop
echo Stopping Climb Analysis. Stored data will be kept.
docker compose stop
if errorlevel 1 goto sub_failed
exit /b 0

:run_restart
echo Restarting Climb Analysis...
docker compose restart
if errorlevel 1 goto sub_failed
echo Climb Analysis is available at http://localhost:8000
exit /b 0

:run_status
docker compose ps
if errorlevel 1 goto sub_failed
exit /b 0

:sub_failed
echo.
echo Command failed. Check that Docker Desktop is running, then try again.
exit /b 1

:help
echo Usage: %~nx0 ^<start^|stop^|restart^|status^>
echo.
echo   start    Build changed images and start the local server
echo   stop     Stop the local server without deleting data
echo   restart  Restart the existing local server
echo   status   Show whether the local server is running
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
