@echo off
setlocal enabledelayedexpansion
rem ============================================================================
rem  assemble.cmd - freeze every BEN spec ONCE (timed), then assemble BENAll.
rem  The BBA/BEN/MvsM assemble scripts only robocopy from dist\ (no rebuilds),
rem  so every spec is built exactly once per BuildAll. Each freeze is wrapped by
rem  the :build subroutine which prints wall-clock seconds so you can see which
rem  step is slow (the TensorFlow exes dominate; Defender scanning their binaries
rem  is the usual culprit - exclude the env + build/dist dirs to speed it up).
rem ============================================================================

for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_ALL0=%%S"

call :build table_manager_client.spec
call :build game.spec
call :build gameapi.spec
call :build gameserver.spec
call :build appserver.spec
call :build TMCGui.spec
call :build BENGui.spec
call :build BEN.spec

for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_ALL1=%%S"
set /a "_ALLEL=_ALL1-_ALL0"
echo.
echo [TIMING] ==== all assemble.cmd freezes done in !_ALLEL!s ====
echo.

echo [assemble] copying dist\ output into BENAll\ ...
robocopy ..\src\config "BENAll\config" /E
robocopy ..\BBA\CC "BENAll\BBA\CC" /E
robocopy ..\models "BENAll\models" /E
robocopy dist\game "BENAll" /E
robocopy dist\gameapi "BENAll" /E
robocopy dist\gameserver "BENAll" /E
robocopy dist\appserver "BENAll" /E
robocopy dist\TMCGUI "BENAll" /E
robocopy dist\BENGUI "BENAll" /E
robocopy dist\BEN "BENAll" /E
robocopy dist\table_manager_client "BENAll" /E
robocopy ..\src\nn "BENAll\nn" *tf2.py*
robocopy ..\bin "BENAll\bin" /E
copy ..\src\ben.ico "BENAll"
copy ..\src\logo.png "BENAll"
goto :eof

:build
rem  %1 = spec file. Times the PyInstaller freeze and prints elapsed seconds.
for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_T0=%%S"
echo.
echo [TIMING] ^>^>^> building %~1 ...
python -m PyInstaller %~1 --noconfirm
set "_RC=!errorlevel!"
for /f %%S in ('powershell -NoProfile -Command "[datetimeoffset]::UtcNow.ToUnixTimeSeconds()"') do set "_T1=%%S"
set /a "_EL=_T1-_T0"
echo [TIMING] ^<^<^< %~1 took !_EL!s  (exit !_RC!)
goto :eof
