conda create -n ben python=3.9

conda activate ben

conda install numpy
conda install scipy
conda install matplotlib
conda install jupyter
conda install scikit-learn
conda install pandas

pip install tqdm

pip install websockets

pip install "tensorflow==2.13"
pip install grpcio-tools

pip install bottle
pip install gevent
pip install pythonnet

# When running ben on ubuntu 22.04, the following error happens during card play.

#OSError: libboost_thread.so.1.71.0: cannot open shared object file: No such file or directory

#Solution: Recompile the double dummy solver library.

#Recompilation will create a new libdds.so file which you need to copy into the bin folder of ben (overwriting the file which is already there)

#In the compilation process of DDS you may have to install the libboost-thread dependency by running the command 

# sudo apt install libboost-thread-dev

# You might also need to install jq if you want to run the automated matches in scripts/match/bidding. 

https://jqlang.github.io/jq/download/


Using chocolatey to install is recommended on windows.
