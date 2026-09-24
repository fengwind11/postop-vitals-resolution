@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
if /I "%~1"=="stage" goto stage
if not "%~1"=="" goto usage
python smoke_test\smoke_test.py
exit /b %errorlevel%
:stage
if "%~2"=="" goto usage
python prepare_private_workspace.py --work-dir "%~2"
exit /b %errorlevel%
:usage
echo Use no arguments for synthetic test, or run_all.cmd stage PRIVATE_WORK_DIRECTORY
exit /b 2
