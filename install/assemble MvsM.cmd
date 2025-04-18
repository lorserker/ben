python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm

robocopy ..\src\config\MvsM "MvsM\config" /E
if not exist "MvsM\config\opponent" mkdir "MvsM\config\opponent"
robocopy ..\src\config\opponent "MvsM\config\opponent" /E
robocopy ..\BBA\CC "MvsM\BBA\CC" /E
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E righty*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E lefty*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Lead-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E dummy_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E decl_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E BEN-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Blue*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E GIB-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Lia-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Q-Plus*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Shark-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Robo-Sayc-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E WBridge5-*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Contract*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E Trick*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E SD_*
robocopy ..\models\TF2Models "MvsM\models\TF2Models" /E RPDD_*
robocopy dist\TMCGUI "MvsM" /E
robocopy dist\table_manager_client "MvsM" /E
robocopy dist\table_manager_client\_internal\bin "MvsM\bin" /E
robocopy ..\src\nn "MvsM\nn" *tf2.py*
robocopy ..\bin "MvsM\bin" /E
copy ..\src\ben.ico "MvsM"
copy ..\src\logo.png "MvsM"