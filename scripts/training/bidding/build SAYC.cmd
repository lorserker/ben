REM No build scrips, but a list of interesting commands

python bidding_binary.py "BBA - 11.03.2024 02.30 SAYC 7949.ben" bin
python bidding_nn.py bin
REM 2024-03-11 15:24:16 180000. c_train=0.09772700816392899
REM 2024-03-11 15:27:47 190000. c_train=0.09641595929861069
REM 2024-03-11 15:31:57 200000. c_train=0.06646695733070374

python binfo_nn.py bin
REM 2024-03-12 14:12:19 220000. c_train=[0.83598125, 0.40536115, 0.4306201]
REM 2024-03-12 14:16:45 230000. c_train=[0.825565, 0.3873856, 0.43817937]
REM 2024-03-12 14:21:00 240000. c_train=[0.7847647, 0.35847667, 0.42628804]

REM Set BEN_HOME first
python testrun.py default159.conf "sampling_1000.ben"

REM 2024-03-12 12:53:33 Matched 378 deals
REM 2024-03-12 12:53:33 better 203 deals
REM 2024-03-12 12:53:33 worse 195 deals
REM 2024-03-12 12:53:33 same score 224 deals

REM For testing the generated model
python DisplayInfoForOpeningBids default159.conf


python bidding_binary.py "BBA - 11.03.2024 02.30 SAYC 7949.ben" "bin NSEW" NS=1 EW=99
python bidding_nn.py "bin NSEW"
REM 2024-03-11 19:33:38 180000. c_train=0.10827173292636871
REM 2024-03-11 19:38:09 190000. c_train=0.11874273419380188
REM 2024-03-11 19:42:42 200000. c_train=0.11186135560274124

python binfo_nn.py bin

REM Set BEN_HOME first
python testrun.py default161.conf "sampling_1000.ben"

REM 2024-03-11 20:10:49 Matched 184 deals
REM 2024-03-11 20:10:49 better 307 deals
REM 2024-03-11 20:10:49 worse 305 deals
REM 2024-03-11 20:10:49 same score 204 deals

REM For testing the generated model
python DisplayInfoForOpeningBids default161.conf

python bidding_coverage.py "BBA - 11.03.2024 02.30 SAYC 7949.ben" "BBA - 10.02.2024 06.20 Random.ben" NS=1 EW=99 
python bidding_nn.py "bidding_bin_same"
REM 2024-03-12 11:41:25 1120000. c_train=0.15812820196151733
REM 2024-03-12 11:45:52 1130000. c_train=0.1401987373828888
REM 2024-03-12 11:50:10 1140000. c_train=0.1606963872909546

python binfo_nn.py "bidding_bin_same"

REM Set BEN_HOME first
python testrun.py default201coverage.conf "sampling_1000.ben"
REM 2024-03-12 12:01:00 Matched 122 deals
REM 2024-03-12 12:01:00 better 318 deals
REM 2024-03-12 12:01:00 worse 292 deals
REM 2024-03-12 12:01:00 same score 268 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default201coverage.conf
