python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller game.spec --noconfirm
python -m PyInstaller gameapi.spec --noconfirm
python -m PyInstaller gameserver.spec --noconfirm
python -m PyInstaller appserver.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm
python -m PyInstaller BENGui.spec --noconfirm
python -m PyInstaller BEN.spec --noconfirm

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