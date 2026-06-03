@echo off
rem ============================================================================
rem  BuildAll.cmd  -  build the BEN release packages.
rem    1. ensure we build from the conda 'ben' env (see requirements.txt)
rem    2. clear the output folders + PyInstaller build/dist cache (no stale files)
rem    3. run the PyInstaller / assemble steps
rem    4. zip each output folder as  <name>-<version>.zip
rem ============================================================================

call "check_env.cmd"
if errorlevel 1 exit /b 1

echo [BuildAll] clearing previous output folders and PyInstaller cache ...
for %%D in (BBA BEN BENAll MvsM build dist) do (
    if exist "%%D\" rmdir /s /q "%%D"
)

call "assemble.cmd"
call "assemble BBA.cmd"
call "assemble BEN.cmd"
call "assemble MvsM.cmd"

rem --- final step: package each output folder as <name>-<version>.zip ----------
set "BEN_VERSION="
for /f "delims=" %%V in ('python "%~dp0_version.py"') do set "BEN_VERSION=%%V"
if not defined BEN_VERSION goto :no_version
if /I "%BEN_VERSION%"=="unknown" goto :no_version

echo [BuildAll] packaging release zips for version %BEN_VERSION% ...
call :zip_folder BBA
if errorlevel 1 exit /b 1
call :zip_folder BEN
if errorlevel 1 exit /b 1
call :zip_folder BENAll
if errorlevel 1 exit /b 1
call :zip_folder MvsM
if errorlevel 1 exit /b 1
echo [BuildAll] done - release zips are in "%CD%"
goto :eof

:zip_folder
if not exist "%~1\" (
    echo [BuildAll]   WARNING: folder "%~1" not found - skipped
    goto :eof
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Compress-Archive -Path '%~1' -DestinationPath '%~1-%BEN_VERSION%.zip' -Force"
if errorlevel 1 (
    echo [BuildAll]   ERROR: failed to create %~1-%BEN_VERSION%.zip
    exit /b 1
)
echo [BuildAll]   created %~1-%BEN_VERSION%.zip
goto :eof

:no_version
echo [BuildAll]   ERROR: could not read the version from ..\src\game.py - zip step skipped
exit /b 1
