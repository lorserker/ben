@echo off
rem --- ensure we build from the conda 'ben' env (see requirements.txt) ---
call "check_env.cmd"
if errorlevel 1 exit /b 1

:: These builds are for windows
:: NOTE: the DDS solver is the 'dds3' Python extension. The .spec files bundle it
:: from the vendored ..\bin\dds3-win (via pathex), so no separate install is
:: needed - but that _dds3.pyd must be built for the build Python (3.12).
pyinstaller "..\src\appserver.py" --onefile --path=..\src --add-data "..\src\frontend;frontend"
pyinstaller "..\src\frontend\appserver.py" --onefile --path=..\src --add-data "..\src\frontend;."  --add-data "..\src\frontend\views;views"

python -m PyInstaller "..\src\pbn2ben.py" --onefile --path=..\src
python -m PyInstaller "..\src\pbn2bba.py" --onefile --path=..\src

:: game / gameserver / gameapi / table_manager_client are built from their .spec
:: files below: those bundle dds3 (pathex ..\bin\dds3-win) and the current
:: bin\BGA\windows\x64 binaries. The old bare "pyinstaller ...py --add-binary
:: ..\bin\BGADLL.dll" lines were removed - that bin path no longer exists and
:: they produced executables without the dds3 solver.

:: calculate_* use the dds3 double-dummy solver -> build from .spec so dds3 is bundled.
python -m PyInstaller calculate_DCWER.spec --noconfirm
python -m PyInstaller calculate_DDOLAR.spec --noconfirm

pyinstaller "..\\scripts\\training\\bidding\\bidding_binary.py" --onefile --path=..\src
pyinstaller "..\\scripts\\training\\bidding\\bidding_nn.py" --onefile --path=..\src
pyinstaller "..\\scripts\\training\\bidding_info\\binfo_binary.py" --onefile --path=..\src
pyinstaller "..\\scripts\\training\\bidding_info\\binfo_nn.py" --onefile --path=..\src


python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller game.spec --noconfirm
python -m PyInstaller gameapi.spec --noconfirm
python -m PyInstaller gameserver.spec --noconfirm
python -m PyInstaller appserver.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm
python -m PyInstaller BENGui.spec --noconfirm
