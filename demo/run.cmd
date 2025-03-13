## Navigate to /src
python game.py --conf config\GIB-BBO.conf --auto True --boards ..\demo\camrose24_1.pbn --outputpbn ..\demo\camrose2024_BEN.pbn --db ..\demo\camrose24 --boardno 1
## Navigate to /src/frontend
ben\src\frontend>python createlog.py --db ..\..\demo\camrose24 
## Copy log.js to /menu