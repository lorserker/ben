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

The input file is set inside the script (currently `../../data/WBC/play.txt`), so edit it there if you want to
train on other data. That file is not in the repository - it is published on the
[training-data release](https://github.com/lorserker/ben/releases/tag/training-data) as
[play_jackformat.zip](https://github.com/lorserker/ben/releases/download/training-data/play_jackformat.zip)
(the same boards in BEN format are in [play_ben.zip](https://github.com/lorserker/ben/releases/download/training-data/play_ben.zip)).

This writes 16 files into the `binary` folder - `X_<seat>_<suit|nt>.npy` and `Y_<seat>_<suit|nt>.npy` for each of `lefty`, `righty`, `decl` and `dummy`. `X` contains the inputs to the neural network and `Y` the expected outputs, both in numpy array format.

Then, run the trainig script. The network is a 3-layer LSTM over the tricks, so the card played can depend on everything seen earlier in the play (signals, show-outs, suits already played), not only on the last trick. The models need Keras 3.15 or later to load.

Use a GPU (on Windows that means WSL): on an RTX 4090 an epoch over the full data takes well under a minute. The last 5% of the deals are held out for validation, training stops when the validation loss stops improving, and a snapshot is saved after every epoch.

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

The first argument is the directory holding the binary files, the second selects which of them to train on. Options can follow as `key=value`: `lstm_size` (128), `n_layers` (3), `epochs` (30), `batch_size` (1024), `learning_rate` (0.001), `keep` (0.8), `val_fraction` (0.05), `patience` (3), `max_examples` (0 = all) and `name`. The generated model is named after that selection plus the date, and is written to the `model` folder; `<name>_<date>-best.keras` is the snapshot with the best validation loss.

To compare models on the held-out deals:

```
python eval_play_nn.py binary righty_suit ../../../models/TF2models/righty_suit_2024-07-16-E20.keras model/righty_suit_<date>-best.keras
```

It prints, per trick and in total, how often the model's top card is the card actually played, and `nll` - how unlikely the model found the card actually played. `nll` is what matters when BEN uses the model to score samples.

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, set the path to the network which you just trained in a configuration file - either `src/config/default.conf` or a new configuration file of your own.

The neural network for leads are using the same principles, but is generating more datasets.

### Making a test run

To test the neural network, it is possible to run game.py and see how the play proceeds
