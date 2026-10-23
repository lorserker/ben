@echo off
setlocal enabledelayedexpansion

REM Let BBA bid all input_*.bba files with one convention card, in parallel, and merge the result
REM Usage: runBBA.cmd CC version [inputdir]
REM   CC        convention card, read from %BEN_HOME%\BBA\CC\<CC>.bbsa
REM   version   BBA version, used in the name of the result: %BEN_HOME%\BBA\Boards\<CC>-<version>.pbn
REM   inputdir  folder with input_1.bba, input_2.bba, ... (default: the current folder)
REM Run it from the BBA folder (where BBA.exe is). Set BEN_HOME if BEN is not in D:\GitHub\ben

if "%~2"=="" (
    echo Error: Please provide a CC name and the BBA version.
    echo Usage: %~nx0 CC version [inputdir]
    exit /b 1
)

set "CC=%~1"
set "version=%~2"
set "inputdir=%~3"
if "!inputdir!"=="" set "inputdir=."
if "%BEN_HOME%"=="" set "BEN_HOME=D:\GitHub\ben"

set "ccfile=!BEN_HOME!\BBA\CC\!CC!.bbsa"
if not exist "!ccfile!" (
    echo Error: Convention card !ccfile! not found
    exit /b 1
)
if not exist BBA.exe (
    echo Error: BBA.exe not found - run this from the BBA folder
    exit /b 1
)

set "merged_file=!BEN_HOME!\BBA\Boards\!CC!-!version!.pbn"
echo Generating !merged_file!

REM Start one BBA instance per input file
set count=0
for %%f in ("!inputdir!\input_*.bba") do (
    set /a count+=1
    set "output_file=!BEN_HOME!\BBA\!CC!_output_!count!"
    del "!output_file!.pbn" /q 2>nul
    start "BBA_!count!" /D . .\BBA.exe --ARCHIVE_FILE "!output_file!" --ARCHIVE_TYPE 4 --AUTOBID --AUTOCLOSE --HAND "%%~ff" --CC1 "!ccfile!" --CC2 "!ccfile!" --DD 0 --SD 1
    echo BBA started for %%~nxf
    ping -n 3 127.0.0.1 >nul
)

if !count!==0 (
    echo Error: No input_*.bba files in !inputdir!
    exit /b 1
)

REM Store the PIDs
set "pids="
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq BBA.exe" /FO CSV /NH') do (
    set "pids=!pids! %%a"
)

REM Wait for all instances to finish
:wait_loop
ping -n 61 127.0.0.1 >nul
set "still_running="

for %%p in (!pids!) do (
    tasklist /FI "PID eq %%~p" | findstr /i "BBA.exe" >nul && set "still_running=1"
)

if defined still_running goto wait_loop

echo All BBA instances finished. Merging files...

REM Merge all output files into a single file
del "!merged_file!" /q 2>nul

for /L %%i in (1,1,!count!) do (
    type "!BEN_HOME!\BBA\!CC!_output_%%i.pbn" >> "!merged_file!"
)

echo Merging complete. Output saved to !merged_file!

endlocal
