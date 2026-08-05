@echo off
setlocal
set "DIR=%~dp0"
python "%DIR%pre-commit" %*
exit /b %ERRORLEVEL%
