rem Must run in anaconda BEN configuration
set BEN_HOME=D:\github\ben\
call match 0.05
call match 0.10
call match 0.20

call vs.cmd 0.20 0.05
call vs.cmd 0.10 0.05
call vs.cmd 0.10 0.20