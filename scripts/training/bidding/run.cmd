
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\GIB-BBO-8730.pbn-OK_boards.pbn       None GIB-BBO       NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25 
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\Robo-Sayc-8730.pbn-OK_boards.pbn     None Robo-Sayc     NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25 
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\BEN-21GF-8730.pbn-OK_boards.pbn      None BEN-21GF      NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\BEN-SAYC-8730.pbn-OK_boards.pbn      None BEN-SAYC      NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\Lia-21GF-8730.pbn-OK_boards.pbn      None Lia-21GF      NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\QPlus-21GF-8730.pbn-OK_boards.pbn    None QPlus-21GF    NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\Robo-Sayc-8730.pbn-OK_boards.pbn     None Robo-Sayc     NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\Shark-SAYC-8730.pbn-OK_boards.pbn    None Shark-SAYC    NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\WBridge5-SAYC-8730.pbn-OK_boards.pbn None WBridge5-SAYC NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\WBridge5-21GF-8730.pbn-OK_boards.pbn None WBridge5-21GF NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25   

python bidding_nn_keras.py BEN-21GF BEN-21GF


REM Testrun with a single deal
python testrun.py default.conf ..\data\1deal.txt
python testrun.py default_tf2.conf ..\data\1deal.txt

REM Testrun with many deals
python testrun.py default.conf ..\data\10000deals.txt
python testrun.py default_tf2.conf ..\data\10000deals.txt
python testrun.py default.conf "..\data\input-2024.02.12.txt"
