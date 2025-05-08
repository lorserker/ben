python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm

if not exist "MvsM\config" mkdir "MvsM\config"
robocopy ..\src\config "MvsM\config" BEN*
robocopy ..\src\config "MvsM\config" BBA*
robocopy ..\src\config "MvsM\config" GIB-BBO.conf*
if not exist "MvsM\config\opponent" mkdir "MvsM\config\opponent"
robocopy ..\src\config\opponent "MvsM\config\opponent" /E
robocopy ..\BBA\CC "MvsM\BBA\CC" /E
robocopy ..\models\TF2Models "MvsM\models\TF2Models" BEN-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" BlueChip-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" GIB-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Lia-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" QPlus-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Shark-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Robo-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" WBridge5-*8730*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" righty*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" lefty*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Lead-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" dummy_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" decl_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Contract*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" Trick*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" SD_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" RPDD_*
robocopy dist\TMCGUI "MvsM" /E
robocopy dist\table_manager_client "MvsM" /E
robocopy dist\table_manager_client\_internal\bin "MvsM\bin" /E
robocopy ..\src\nn "MvsM\nn" *tf2.py*
robocopy ..\bin "MvsM\bin" /E
copy ..\src\ben.ico "MvsM"
copy ..\src\logo.png "MvsM"