cd frontend
start python appserver.py --host 0.0.0.0
cd ..
start "App Server" python appserver.py --port 8081
start "TF2 Api :80" python gameapi.py --config config\default_TF2.conf --port 80 --host 0.0.0.0
start "TF2 Api :8085" python gameapi.py --config config\default_TF2.conf --host 0.0.0.0
start "Old Api :8099" python gameapi.py --host 0.0.0.0 --config config\default_Api.conf --port 8088
start "2/1" python gameserver.py --config config\default_api.conf --port 4440
start "SAYC" python gameserver.py --config config\SAYC.conf --port 4441
start "SAYC UCBC" python gameserver.py --config "..\UCBC 2024\Conf\UCBC2024src.conf" --port 4442
start "Jackos" python gameserver.py --config config\muppet.conf --port 4443
start "BBA" python gameserver.py --config config\BBA-2over1.conf --port 4444
start "Default" python gameserver.py --config config\default.conf --port 4445
start "Jack Sayc" python gameserver.py --config config\jacksayc.conf --port 4446
start "TF2 GIB" python gameserver.py --config config\default_TF2.conf --port 4447
