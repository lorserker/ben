rem Must run in anaconda BEN configuration
set BEN_HOME=D:\github\ben\
call match 0.05 default set0010.txt gamedb1
call match 0.10 default set0010.txt gamedb2
call match 0.20 default set0010.txt gamedb3

call vs.cmd 0.20 0.05 default set0010.txt gamedb4
call vs.cmd 0.10 0.05 default set0010.txt gamedb5
call vs.cmd 0.10 0.20 default set0010.txt gamedb6