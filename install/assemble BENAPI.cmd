rem gameapi is already built into dist\ by assemble.cmd (which BuildAll.cmd runs
rem first), so this script only robocopies the existing dist\ output into the
rem BENAPI\ layout. BENAPI is the API-only package: gameapi.exe and nothing else
rem executable - no gameserver, no appserver, no GUI, no table manager client.
rem The config/model selection is the same as MvsM (the 8730 systems), plus
rem default_api.conf - that is what gameapi.exe loads when started without
rem --config, and none of the MvsM patterns match it.

if not exist "BENAPI\config" mkdir "BENAPI\config"
robocopy ..\src\config "BENAPI\config" BEN*
robocopy ..\src\config "BENAPI\config" BBA*
robocopy ..\src\config "BENAPI\config" GIB-BBO.conf*
robocopy ..\src\config "BENAPI\config" default_api.conf
rem systems.conf: the systems a request may ask for (ourSystem/theirSystem)
robocopy ..\src\config "BENAPI\config" systems.conf
if not exist "BENAPI\config\opponent" mkdir "BENAPI\config\opponent"
robocopy ..\src\config\opponent "BENAPI\config\opponent" /E
robocopy ..\BBA\CC "BENAPI\BBA\CC" /E
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" BEN-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" BEN-*8742*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" BlueChip-*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" GIB-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" GIB-*8742*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Lia-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" QPlus-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Shark-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Robo-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" WBridge5-*8730*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" righty*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" lefty*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Lead*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" dummy_*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" decl_*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Contract*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" Trick*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" SD_*
robocopy ..\models\TF2Models "BENAPI\models\TF2Models" RPDD_*
robocopy dist\gameapi "BENAPI" /E
robocopy ..\src\nn "BENAPI\nn" *tf2.py* keras_compat.py
robocopy ..\bin "BENAPI\bin" /E
copy ..\src\ben.ico "BENAPI"
copy ..\src\logo.png "BENAPI"
