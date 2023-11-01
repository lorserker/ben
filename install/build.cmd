:: These builds are for windows
pyinstaller ..\src\appserver.py --onefile --path=..\src --add-data ..\src\frontend;frontend
pyinstaller ..\src\game.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.
pyinstaller ..\src\gameserver.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.
pyinstaller ..\src\table_manager_client.py --onefile --path=..\src --add-binary ..\bin\dds.dll;. 
pyinstaller ..\src\calculate_DCWER.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.
pyinstaller ..\src\calculate_DDOLAR.py --onefile --path=..\src --add-binary ..\bin\dds.dll;.

pyinstaller ..\\scripts\\training\\bidding\\bidding_binary.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding\\bidding_nn.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding_info\\binfo_binary.py --onefile --path=..\src 
pyinstaller ..\\scripts\\training\\bidding_info\\binfo_nn.py --onefile --path=..\src 
