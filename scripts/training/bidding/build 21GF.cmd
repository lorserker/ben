REM No build scripts, but a list of interesting commands

REM Build set of 1000 board to use for validation
python select_sample.py "BBA - 11.03.2024 09.56 21GF 7949.ben"

REM The original BEN
python bidding_binary.py "BBA - 11.03.2024 09.56 21GF 7949.ben" "bin"
python bidding_nn.py "bin"
REM 2024-03-12 18:01:03 180000. c_train=0.10988911986351013
REM 2024-03-12 18:05:27 190000. c_train=0.09024004638195038
REM 2024-03-12 18:09:51 200000. c_train=0.10825099050998688

python binfo_nn.py "bin"

REM Set BEN_HOME first
python testrun.py default159.conf "sampling_1000.ben"

REM 2024-03-12 20:37:01 Matched 381 deals
REM 2024-03-12 20:37:01 better 201 deals
REM 2024-03-12 20:37:01 worse 222 deals
REM 2024-03-12 20:37:01 same score 196 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default159.conf

REM The different system for NS and EW
python bidding_binary.py "BBA - 11.03.2024 09.56 21GF 7949.ben" "bin NSEW" NS=1 EW=99
python bidding_nn.py "bin NSEW"
REM 2024-03-11 19:33:38 180000. c_train=0.10827173292636871
REM 2024-03-11 19:38:09 190000. c_train=0.11874273419380188
REM 2024-03-11 19:42:42 200000. c_train=0.11186135560274124

python binfo_nn.py "bin NSEW"

REM Set BEN_HOME first
python testrun.py default161.conf "sampling_1000.ben"

REM 2024-03-11 20:10:49 Matched 184 deals
REM 2024-03-11 20:10:49 better 307 deals
REM 2024-03-11 20:10:49 worse 305 deals
REM 2024-03-11 20:10:49 same score 204 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default161.conf

REM Same system for N and S
python bidding_binary.py "BBA - 11.03.2024 09.56 21GF 7949.ben" "bin NS Same" NS=1 EW=99
python bidding_nn.py "bin NS Same" 
REM 2024-03-14 16:05:32 480000. c_train=0.14426739513874054
REM 2024-03-14 16:09:19 490000. c_train=0.10429387539625168
REM 2024-03-14 16:13:48 500000. c_train=0.15810474753379822

python binfo_nn.py "bin NS Same" 

REM Set BEN_HOME first
python testrun.py default161same.conf "sampling_1000.ben"

REM 2024-03-11 22:21:51 Matched 182 deals
REM 2024-03-11 22:21:51 better 309 deals
REM 2024-03-11 22:21:51 worse 296 deals
REM 2024-03-11 22:21:51 same score 213 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default161same.conf

REM Included 4 bids in the input
python bidding_binary.py "BBA - 11.03.2024 09.56 21GF 7949.ben" "bin V2" NS=1 EW=99 version=2
python bidding_nn.py "bin V2"
REM 2024-03-11 23:35:31 180000. c_train=0.11700223386287689
REM 2024-03-11 23:39:43 190000. c_train=0.13824942708015442
REM 2024-03-11 23:43:45 200000. c_train=0.12440584599971771
python binfo_nn.py "bin V2"
REM 2024-03-13 22:04:31 180000. c_train=[0.81523454, 0.36946273, 0.44577184]
REM 2024-03-13 22:08:15 190000. c_train=[0.8343884, 0.40360963, 0.43077874]
REM 2024-03-13 22:12:41 200000. c_train=[0.8628615, 0.42583546, 0.43702602]

REM Set BEN_HOME first
python testrun.py default201.conf "sampling_1000.ben"
REM 2024-03-13 22:48:31 Matched 328 deals
REM 2024-03-13 22:48:31 better 213 deals
REM 2024-03-13 22:48:31 worse 256 deals
REM 2024-03-13 22:48:31 same score 203 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default201.conf

REM Reducing overrepresented sequences and adding boards to fill the gaps
python bidding_coverage.py "BBA - 11.03.2024 09.56 21GF 7949.ben" "BBA - 11.03.2024 13.04 GF21 7949" NS=1 EW=99
python bidding_nn.py "bidding_bin_same"
REM 2024-03-12 11:41:25 1120000. c_train=0.15812820196151733
REM 2024-03-12 11:45:52 1130000. c_train=0.1401987373828888
REM 2024-03-12 11:50:10 1140000. c_train=0.1606963872909546

python binfo_nn.py "bidding_bin_same"

REM Set BEN_HOME first
python testrun.py default201coverage.conf "BBA - 11.03.2024 09.56 21GF 7949.ben"
REM 2024-03-12 12:01:00 Matched 122 deals
REM 2024-03-12 12:01:00 better 318 deals
REM 2024-03-12 12:01:00 worse 292 deals
REM 2024-03-12 12:01:00 same score 268 deals

REM For testing the generated model
python DisplayInfoForOpeningBids.py default201coverage.conf
