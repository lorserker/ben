REM No build scrips, but a list of interesting commands

python bidding_coverage.py '..\data\BBA - 02.02.2024 13.29.ben' '..\data\5 mill deals with bidding.ben' NS=1 EW=99 >stats.txt
python bidding_binary.py ../data/bidding_data/bidding_data.txt bidding_bin 

python bidding_coverage.py "..\data\BBA SAYC vs SAYC - 20.02.2024 02.52.ben" "..\data\BBA SAYC vs SAYC - 20.02.2024 02.52.ben" NS=1 EW=99 >stats.txt 

python bidding_nn.py bidding_bin_same 
python bidding_and_info_nn.py bidding_bin_same 
python bidding_nn_tf2.py bidding_bin 

REM Testrun with a single deal
python testrun.py default.conf ..\data\1deal.txt
python testrun.py default_tf2.conf ..\data\1deal.txt

REM Testrun with many deals
python testrun.py default.conf ..\data\10000deals.txt
python testrun.py default_tf2.conf ..\data\10000deals.txt
python testrun.py default.conf "..\data\input-2024.02.12.txt"
