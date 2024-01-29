:: These builds are for windows
pyinstaller ..\src\appserver.py --onefile --path=..\src --add-data "..\src\frontend;frontend"
pyinstaller ..\src\frontend\appserver.py --onefile --path=..\src --add-data "..\src\frontend;."  --add-data "..\src\frontend\views;views"

pyinstaller ..\src\pbn2ben.py --onefile --path=..\src 
pyinstaller ..\src\pbn2bba.py --onefile --path=..\src 

pyinstaller ..\src\game.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.
pyinstaller ..\src\gameserver.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.
pyinstaller ..\src\table_manager_client.py --onefile --path=..\src --add-binary ..\bin\dds.dll;. 
pyinstaller "..\\scripts\\training\\playing\\calculate_DCWER.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."
pyinstaller "..\\scripts\\training\\opening lead\\calculate_DDOLAR.py" --onefile --path=..\src --add-binary "..\bin\dds.dll;."

pyinstaller ..\\scripts\\training\\bidding\\bidding_binary.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding\\bidding_nn.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding_info\\binfo_binary.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding_info\\binfo_nn.py --onefile --path=..\src 
