@echo off
setlocal enabledelayedexpansion

REM Check if the parameter is provided
if "%~1"=="" (
    echo Error: Please provide a CC name as a parameter.
    echo Usage: %~nx0 FileName
    exit /b 1
)

REM Store the passed parameter
set "CC=%~1"
set "version=8730"

REM Merge all output files into a single file
set "merged_file=D:\GitHub\ben\BBA\Boards\!CC!-!version!.pbn"
echo Generating !merged_file!


for /L %%i in (1,1,6) do (
    set "input_file=.\input_%%i.bba"
    set "output_file=D:\GitHub\ben\BBA\!CC!_output_%%i"
    del D:\GitHub\ben\BBA\!CC!_output_%%i.pbn /q
    start "BBA_%%i" /D . BBA.exe --ARCHIVE_FILE !output_file!  --ARCHIVE_TYPE 4 --AUTOBID --AUTOCLOSE --HAND !input_file! --CC1 "D:\GitHub\ben\BBA\CC\!CC!.bbsa" --CC2 "D:\GitHub\ben\BBA\CC\!CC!.bbsa" --DD 0 --SD 1"
    
    REM Store the PID
    for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq BBA.exe" /FO CSV /NH') do (
        set "pids=!pids! %%a"
    )    
    echo BBA Started for input file %%i
    timeout /t 2 /nobreak >nul
)

REM Wait for all instances to finish
:wait_loop
timeout /t 60 /nobreak >nul
set "still_running="

for %%p in (%pids%) do (
    tasklist /FI "PID eq %%p" | findstr /i "BBA.exe" >nul && set "still_running=1"
)

if defined still_running goto wait_loop

echo All BBA instances finished. Merging files...

REM Merge all output files into a single file
set "merged_file=D:\GitHub\ben\BBA\Boards\!CC!-!version!.pbn"
del "!merged_file!" /q

for /L %%i in (1,1,6) do (
    type "D:\GitHub\ben\BBA\!CC!_output_%%i.pbn" >> "!merged_file!"
)

echo Merging complete. Output saved to !merged_file!

endlocal
