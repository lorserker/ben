python ..\..\training\bidding\bidding_binary.py training.txt ..\..\..\models\withsystem\SAYC\binary NS=1 EW=1

python ..\..\training\bidding\bidding_nn.py ..\..\..\models\withsystem\SAYC\binary ..\..\..\models\withsystem\SAYC

python ..\..\training\bidding\binfo_binary.py training.txt ..\..\..\models\withsystem\SAYC\binary_info NS=1 EW=1

python ..\..\training\bidding\binfo_nn.py ..\..\..\models\withsystem\SAYC\binary_info ..\..\..\models\withsystem\SAYC
