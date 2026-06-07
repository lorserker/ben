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

rem  Clear the OUTPUT folders + dist (so no stale/test artifacts ship), but KEEP
rem  the PyInstaller 'build\' cache: its Analysis-00.toc lets an unchanged spec
rem  skip the slow binary-reclassification / dynamic-library scan entirely. Run
rem  the app from dist\, never from build\, so build\ stays a pure cache.
echo [BuildAll] clearing previous output folders + dist (keeping build\ cache) ...
for %%D in (BBA BEN BENAll MvsM dist) do (
    if exist "%%D\" rmdir /s /q "%%D"
)

rem --- timed phases (epoch seconds -> locale-independent elapsed) --------------
for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_BUILD0=%%S"

call :phase "assemble.cmd"
call :phase "assemble BBA.cmd"
call :phase "assemble BEN.cmd"
call :phase "assemble MvsM.cmd"

for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_BUILD1=%%S"
set /a "_BUILDEL=_BUILD1-_BUILD0"
echo.
echo [TIMING] ======== all freeze+assemble phases done in %_BUILDEL%s ========
echo.

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

:phase
rem  %1 = assemble script to call. Prints wall-clock seconds for the phase.
for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_P0=%%S"
echo.
echo [TIMING] ######## phase START: %~1 ########
call "%~1"
for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_P1=%%S"
set /a "_PEL=_P1-_P0"
echo [TIMING] ######## phase DONE : %~1  in %_PEL%s ########
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
