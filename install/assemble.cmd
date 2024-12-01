python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller game.spec --noconfirm
python -m PyInstaller gameapi.spec --noconfirm
python -m PyInstaller gameserver.spec --noconfirm
python -m PyInstaller appserver.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm
python -m PyInstaller BENGui.spec --noconfirm

robocopy ..\src\config "BEN\config" /E
robocopy ..\BBA\CC "BEN\BBA\CC" /E
robocopy ..\models "BEN\models" /E 
robocopy dist\game "BEN" /E
robocopy dist\gameapi "BEN" /E
robocopy dist\gameserver "BEN" /E
robocopy dist\appserver "BEN" /E
robocopy dist\TMCGUI "BEN" /E
robocopy dist\BENGUI "BEN" /E
robocopy dist\table_manager_client "BEN" /E
robocopy ..\src\nn "BEN\nn" *tf2.py*