:: These builds are for windows
pyinstaller "..\src\appserver.py" --onefile --path=..\src --add-data "..\src\frontend;frontend"
pyinstaller "..\src\frontend\appserver.py" --onefile --path=..\src --add-data "..\src\frontend;."  --add-data "..\src\frontend\views;views"

python -m PyInstaller "..\src\pbn2ben.py" --onefile --path=..\src 
python -m PyInstaller "..\src\pbn2bba.py" --onefile --path=..\src 

pyinstaller "..\src\game.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."  --add-binary "..\bin\BGADLL.dll;." --add-binary "..\bin\libbcalcdds.dll;."
pyinstaller "..\src\gameserver.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."  --add-binary "..\bin\BGADLL.dll;." --add-binary "..\bin\libbcalcdds.dll;."
pyinstaller "..\src\table_manager_client.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."  --add-binary "..\bin\BGADLL.dll;." --add-binary "..\bin\libbcalcdds.dll;."
pyinstaller "..\\scripts\\training\\playing\\calculate_DCWER.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."
pyinstaller "..\\scripts\\training\\opening lead\\calculate_DDOLAR.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."

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
