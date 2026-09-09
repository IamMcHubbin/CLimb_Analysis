@echo off
setlocal
pushd "%~dp0"

if "%~1"=="" goto help
if /I "%~1"=="start" goto start
if /I "%~1"=="stop" goto stop
if /I "%~1"=="restart" goto restart
if /I "%~1"=="status" goto status
goto help

:start
echo Starting Climb Analysis and rebuilding changed images...
docker compose up -d --build
if errorlevel 1 goto failed
echo Climb Analysis is available at http://localhost:8000
goto done

:stop
echo Stopping Climb Analysis. Stored data will be kept.
docker compose stop
if errorlevel 1 goto failed
goto done

:restart
echo Restarting Climb Analysis...
docker compose restart
if errorlevel 1 goto failed
echo Climb Analysis is available at http://localhost:8000
goto done

:status
docker compose ps
if errorlevel 1 goto failed
goto done

:help
echo Usage: %~nx0 ^<start^|stop^|restart^|status^>
echo.
echo   start    Build changed images and start the local server
echo   stop     Stop the local server without deleting data
echo   restart  Restart the existing local server
echo   status   Show whether the local server is running
echo.
echo Run this file from Command Prompt, or create a shortcut whose target is:
echo   "%~f0" start
goto done

:failed
echo.
echo Command failed. Check that Docker Desktop is running, then try again.
popd
exit /b 1

:done
popd
exit /b 0
