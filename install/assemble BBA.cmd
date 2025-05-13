python -m PyInstaller table_manager_client.spec --noconfirm
python -m PyInstaller TMCGui.spec --noconfirm

robocopy ..\BBA\CC "BBA\BBA\CC" Acol.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" BBA-21GF.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" BBA-SAYC.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" BEN-21GF.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" BEN-Sayc.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" BlueChip-Sayc.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" GIB-21GF.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" GIB-BBO.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" Lia-21GF.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" Micro Bridge 13.4.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" QPlus-21GF.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" Robo-Sayc.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" Shark-Sayc.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" WBridge5-Sayc.bbsa
robocopy ..\BBA\CC "BBA\BBA\CC" WJ.bbsa

robocopy ..\src\config\opponent "BBA\config\opponent" /E

robocopy ..\src\config "BBA\config" BBA-21GF.conf
robocopy ..\src\config "BBA\config" BBA-SAYC.conf
robocopy ..\src\config "BBA\config" BEN-21GF.conf
robocopy ..\src\config "BBA\config" BEN-Sayc.conf
robocopy ..\src\config "BBA\config" GIB-BBO.conf

robocopy ..\src\config\Robots "BBA\config" BlueChip-Sayc.conf
robocopy ..\src\config\Robots "BBA\config" GIB-21GF.conf
robocopy ..\src\config\Robots "BBA\config" Lia-21GF.conf
robocopy ..\src\config\Robots "BBA\config" QPlus-21GF.conf
robocopy ..\src\config\Robots "BBA\config" Robo-Sayc.conf
robocopy ..\src\config\Robots "BBA\config" Shark-Sayc.conf
robocopy ..\src\config\Robots "BBA\config" WBridge5-Sayc.conf

robocopy ..\models\TF2Models "BBA\models\TF2Models" /E righty*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E lefty*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Lead-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E dummy_*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E decl_*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E BEN-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Blue*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E GIB-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Lia-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Q-Plus*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Shark-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Robo-Sayc-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E WBridge5-*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E SD_*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E RPDD_*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Contract*
robocopy ..\models\TF2Models "BBA\models\TF2Models" /E Trick*
robocopy dist\TMCGUI "BBA" /E
robocopy dist\table_manager_client "BBA" /E
robocopy dist\table_manager_client\_internal\bin "BBA\bin" /E
robocopy ..\src\nn "BBA\nn" *tf2.py*
robocopy ..\bin "BBA\bin" /E
copy ..\src\ben.ico "BBA"
copy ..\src\logo.png "BBA"