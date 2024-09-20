@echo off
setlocal enabledelayedexpansion

for /L %%i in (1,1,6) do (
    set "input_file=.\input_%%i.bba"
    set "output_file=output_%%i"
    
    set "command=start "BBA" /D . BBA.exe --CURRENT_ARCHIVE . --ARCHIVE_FILE !output_file! --AUTOBID --HAND !input_file! --CC1 GIB-Thorvald.bbsa --CC2 GIB-Thorvald.bbsa --DD 0 --SD 1"
    echo !command!
    
    !command!
    echo BBA Started for input file %%i
    timeout /t 2 /nobreak >nul
)

endlocal
