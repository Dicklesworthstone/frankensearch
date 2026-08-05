@echo off
setlocal
set "DIR=%~dp0"
python "%DIR%pre-push" %*
exit /b %ERRORLEVEL%
