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
# Had to install : conda install -c conda-forge websockets

pip install "tensorflow>=2.0"
pip install grpcio-tools

pip install bottle
pip install gevent

# Seems the wrong version of tensorflow-estimator is being installed, so this might be needed
conda install tensorflow-estimator=1.15