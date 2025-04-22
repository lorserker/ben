python -m PyInstaller gameserver.spec --noconfirm
python -m PyInstaller appserver.spec --noconfirm
python -m PyInstaller BEN.spec --noconfirm

robocopy ..\models\TF2Models "BEN\models\TF2Models" /E righty*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E lefty*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E Lead-*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E dummy_*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E decl_*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E SD_*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E RPDD_*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E Contract*
robocopy ..\models\TF2Models "BEN\models\TF2Models" /E Trick*
if not exist "BEN\config" mkdir "BEN\config"
if not exist "BEN\config\opponent" mkdir "BEN\config\opponent"
robocopy BEN\config\opponent "BEN\config\opponent" /E
copy ..\src\config\BEN-Sayc.conf "BEN\config\default.conf" /Y
if not exist "BEN\BBA" mkdir "BEN\BBA"
if not exist "BEN\BBA\CC" mkdir "BEN\BBA\CC"
copy ..\BBA\CC\BEN-SAYC.bbsa "BEN\BBA\CC\BEN-SAYC.bbsa" /Y
copy ..\models\TF2models\BEN-Sayc-8730_2025-04-20-E30.keras "BEN\models\TF2Models" /Y 
copy ..\models\TF2models\BEN-Sayc-Info-8730_2025-04-20-E30.keras "BEN\models\TF2Models" /Y
copy ..\src\logo.png "BEN"
copy ..\src\ben.ico "BEN"
robocopy dist\gameserver "BEN" /E
robocopy dist\appserver "BEN" /E
robocopy dist\BEN "BEN" /E
robocopy ..\src\nn "BEN\nn" *tf2.py*
robocopy ..\bin "BEN\bin" /E