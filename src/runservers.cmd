cd frontend
start "App Server - Local" python appserver.py --host 0.0.0.0
cd ..
timeout /t 10 /nobreak >nul
start "App Server - Local on port :8081" python appserverold.py --port 8081
timeout /t 10 /nobreak >nul
rem Used from BBO 
start "TF2 Api external :80 (BBO)" python gameapi.py --config config\GIB-BBO.conf --port 80 --host 0.0.0.0
timeout /t 10 /nobreak >nul
rem start "TF2 Api internal :8085" python gameapi.py --config config\default_TF2.conf --host 0.0.0.0
start "TF2 Api internal :8085 (API)" python gameapi.py --config config\GIB-BBO.conf --host 0.0.0.0
timeout /t 10 /nobreak >nul
start "BEN 2/1" python gameserver.py --config config\BEN-21GF.conf --port 4440
timeout /t 10 /nobreak >nul
start "BEN SAYC" python gameserver.py --config config\BEN-SAYC.conf --port 4441
timeout /t 10 /nobreak >nul
start "TF2 GIB" python gameserver.py --config config\GIB-BBO.conf --port 4442
timeout /t 10 /nobreak >nul
start "Default (GIB-BBO)" python gameserver.py --config config\default.conf --port 4443
timeout /t 10 /nobreak >nul
start "BBA 2/1" python gameserver.py --config config\BBA-21GF.conf --port 4444
timeout /t 10 /nobreak >nul
start "BBA Sayc" python gameserver.py --config config\BBA-Sayc.conf --port 4445
