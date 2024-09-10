cd frontend
start "App Server - Local" python appserver.py --host 0.0.0.0
cd ..
timeout /t 2 /nobreak >nul
start "App Server - Local on port :8081" python appserver.py --port 8081
timeout /t 2 /nobreak >nul
rem Used from BBO 
start "TF2 Api external :80 (BBO)" python gameapi.py --config config\default_TF2_Short.conf --port 80 --host 0.0.0.0
timeout /t 10 /nobreak >nul
rem start "TF2 Api internal :8085" python gameapi.py --config config\default_TF2.conf --host 0.0.0.0
start "TF2 Api internal :8085 (API)" python gameapi.py --config config\default_TF2_Short.conf --host 0.0.0.0
timeout /t 10 /nobreak >nul
start "Old Api :8088" python gameapi.py --host 0.0.0.0 --config config\default_Api.conf --port 8088
timeout /t 2 /nobreak >nul
start "2/1" python gameserver.py --config config\default_api.conf --port 4440
timeout /t 2 /nobreak >nul
start "SAYC" python gameserver.py --config config\SAYC.conf --port 4441
timeout /t 2 /nobreak >nul
start "SAYC UCBC" python gameserver.py --config "..\UCBC 2024\Conf\UCBC2024src.conf" --port 4442
timeout /t 2 /nobreak >nul
start "Jackos" python gameserver.py --config config\muppet.conf --port 4443
timeout /t 2 /nobreak >nul
start "BBA" python gameserver.py --config config\BBA-2over1.conf --port 4444
timeout /t 2 /nobreak >nul
start "Default" python gameserver.py --config config\default.conf --port 4445
timeout /t 2 /nobreak >nul
start "Jack Sayc" python gameserver.py --config config\jacksayc.conf --port 4446
timeout /t 2 /nobreak >nul
start "TF2 GIB" python gameserver.py --config config\default_TF2.conf --port 4447
timeout /t 2 /nobreak >nul
start "TF2 Jackos" python gameserver.py --config config\muppet_TF2.conf --port 4448
