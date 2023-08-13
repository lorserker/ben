rem Must run in anaconda BEN configuration
set BEN_HOME=D:\github\ben\
call match 0.05
call match 0.1
call match 0.2

call vs.cmd 0.2 0.05
call vs.cmd 0.1 0.05
call vs.cmd 0.1 0.2