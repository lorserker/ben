REM No build scrips, but a list of interesting commands

python bidding_coverage.py '..\data\BBA - 29.12.2023 17.30.ben' '..\data\5 mill deals with bidding.ben' NS=1 EW=99 alternate=True
python bidding_binary.py ../data/bidding_data/bidding_data.txt bidding_bin 

python bidding_nn.py bidding_bin 
python bidding_nn_tf2.py bidding_bin 

REM Testrun with a single deal
python testrun.py default.conf ..\data\1deal.ben
python testrun.py default_tf2.conf ..\data\1deal.ben

REM Testrun with many deals
python testrun.py default.conf ..\data\10000deals.ben
python testrun.py default_tf2.conf ..\data\10000deals.ben
