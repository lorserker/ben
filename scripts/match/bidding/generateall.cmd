rem Must run in anaconda BEN configuration
set BEN_HOME=D:\github\ben\
call match 0.05 set0010.txt
call match 0.10 set0010.txt
call match 0.20 set0010.txt

call vs.cmd 0.20 0.05 set0010.txt
call vs.cmd 0.10 0.05 set0010.txt
call vs.cmd 0.10 0.20 set0010.txt