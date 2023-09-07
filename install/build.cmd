:: These builds are for windows
pyinstaller ..\src\appserver.py --onefile --path=..\src --add-data ..\src\frontend;frontend
pyinstaller ..\src\game.py --onefile --path=..\src --add-binary C:\\Python311\\Lib\\site-packages\\endplay\\_dds\\dds.dll;.   
pyinstaller ..\src\gameserver.py --onefile --path=..\src --add-binary C:\\Python311\\Lib\\site-packages\\endplay\\_dds\\dds.dll;.   
pyinstaller ..\src\table_manager_client.py --onefile --path=..\src --add-binary C:\\Python311\\Lib\\site-packages\\endplay\\_dds\\dds.dll;.   