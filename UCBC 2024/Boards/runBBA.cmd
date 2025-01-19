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

for /L %%i in (1,1,6) do (
    set "input_file=.\input_%%i.bba"
    set "output_file=D:\GitHub\ben\BBA\output_%%i"
    del D:\GitHub\ben\BBA\output_%%i.pbn /q
    set "command=start "BBA" /D . BBA.exe --ARCHIVE_FILE !output_file! --AUTOBID --HAND !input_file! --CC1 "D:\GitHub\ben\BBA\CC\!CC!.bbsa" --CC2 "D:\GitHub\ben\BBA\CC\!CC!.bbsa" --DD 0 --SD 1"
    echo !command!
    !command!
    echo BBA Started for input file %%i
    timeout /t 2 /nobreak >nul
)

endlocal
