# Training a Neural Network to play

When playing we are using 5 different neural networks in a combination with a double dummy solver.

We have one neural network for finding the opening lead, and a neural network to find the actual play for each position, but this is only used if the there are equal cards from the Double Summy solver.

The NN for each position is to find:

Agreements like top-of-sequence, high from doubleton, top-of-nothing, etc. are picked up by the neural networks if they are in the data.
Also general patterns like "second hand low", "third hand high", "cover the honor", "draw trumps", "finesse" are picked up by the cardplay NN.



### Instructions

You need to be located in the `scripts/training/playing` directory when you execute the following steps.

First, activate the environment

```
conda activate ben
```

You should have a `playingdata.txt` (not included in this repository) file now which contains deals and their corresponding auctions like this:

```
W:AT73.JT4.92.A984 K4.AK76.KQT85.T2 QJ9865.Q53.64.65 2.982.AJ73.KQJ73
W ALL 3N.-1.N
PP 1N PP 3N PP PP PP
SQS2SAS4S3SKS5H2D5D4DJD2H9H4HAH3DTD6D3D9D8S6D7HTDQC5DAC4CKCAC2C6STH6S8C3S7H7S9C7SJH8C8CTHQCJHJHKDKH5CQC9
W:A653.987.8752.74 942.T32.Q43.AKJ3 KQJ.KQ6.KJ6.QT98 T87.AJ54.AT9.652
E EW 1N.-1.E
1N PP PP PP
H4H7HTHKSKS8S3S9SJS7S5S2SQSTSAS4S6H2C9C6C4CKC8C5H3HQHAH8HJH9D3H6H5D2C3D6C2C7CACTCJCQD9D5DJDTD7DQD4DKDAD8
...
```
Run the script to transform the data into binary format. A single script now covers all four seats, and splits each of them into a suit variant and a NT variant:

```
python binary_keras_suit_nt.py
```

The input file is set inside the script (currently `../../data/WBC/play.txt`), so edit it there if you want to train on other data.

This writes 16 files into the `binary` folder - `X_<seat>_<suit|nt>.npy` and `Y_<seat>_<suit|nt>.npy` for each of `lefty`, `righty`, `decl` and `dummy`. `X` contains the inputs to the neural network and `Y` the expected outputs, both in numpy array format.

Then, run the trainig script. This will take several hours to complete, but it will save snapshots of the model as it progresses. If you have a GPU, the training will run faster, but not much faster, because GPUs are not so well suited for the type of NN used.

```
python play_nn_keras.py binary decl_nt
python play_nn_keras.py binary decl_suit
python play_nn_keras.py binary dummy_nt
python play_nn_keras.py binary dummy_suit
python play_nn_keras.py binary lefty_nt
python play_nn_keras.py binary lefty_suit
python play_nn_keras.py binary righty_nt
python play_nn_keras.py binary righty_suit
```

The first argument is the directory holding the binary files, the second selects which of them to train on. The generated model is named after that selection plus the date, and is written to the `model` folder.

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, set the path to the network which you just trained in a configuration file - either `src/config/default.conf` or a new configuration file of your own.

The neural network for leads are using the same principles, but is generating more datasets.

### Making a test run

To test the neural network, it is possible to run game.py and see how the play proceeds
