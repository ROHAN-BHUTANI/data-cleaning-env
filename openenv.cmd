@echo off
setlocal

if /I "%1"=="validate" (
  py "%~dp0openenv_validate.py"
  exit /b %ERRORLEVEL%
)

echo openenv validate failed: unsupported command
exit /b 1
