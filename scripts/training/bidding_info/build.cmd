python binfo_binary.py ../data/Jack/BW5C_Total.ben binfo_bin 
python binfo_nn.py binfo_bin model
python binfo_nn_tf2.py binfo_bin model

python binfo_nn.py ..\bidding\bidding_bin\binfo_bin model

REM For testing the generated model
python DisplayInfoForOpeningBids default.conf
python DisplayInfoForOpeningBids default_tf2.conf

