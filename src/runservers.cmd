cd frontend
start python appserver.py --host 0.0.0.0
cd ..
start python appserver.py --port 8081
start python gameapi.py --config config\default_TF2.conf --port 80 --host 0.0.0.0
start python gameapi.py --config config\default_TF2.conf --host 0.0.0.0
start python gameapi.py --host 0.0.0.0 --config config\default_Api.conf --port 8088
start python gameserver.py --config config\default_api.conf --port 4440
start python gameserver.py --config config\SAYC.conf --port 4441
start python gameserver.py --config "..\UCBC 2024\Conf\UCBC2024src.conf" --port 4442
start python gameserver.py --config config\muppet.conf --port 4443
start python gameserver.py --config config\BBA-2over1.conf --port 4444
start python gameserver.py --config config\default.conf --port 4445
start python gameserver.py --config config\jacksayc.conf --port 4446
start python gameserver.py --config config\default_TF2.conf --port 4447
