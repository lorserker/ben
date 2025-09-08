# Training a Neural Network to find opening lead

### Instructions

You need to be located in the `scripts/training/opening lead` directory when you execute the following steps.

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
Run the script to transform the data into binary format.
You have these scripts available

```
python binary_lead_keras_suit_nt.py 
```

The above command will create new files into the corresponding folder: `A.npy`, `B.npy`, `x.npy` and `y.npy` . `A.npy`, `B.npy`, `x.npy` contains the inputs to the neural network and `y.npy` contains the expected outputs. Both are stored in numpy array format.

Then, run the trainig script. This will take several hours to complete, but it will save snapshots of the model as it progresses. If you have a GPU, the training will run faster, but not much faster, because GPUs are not so well suited for the type of NN used.

```
python lead_nn_keras.py 
```

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, edit the configuration file ([code here](https://github.com/lorserker/ben/blob/main/src/config/default.conf)) inserting the path to the network which you just trained.

The neural network for leads are using the same principles, but is generating more datasets.

### Making a test run

To test the neural network, it is possible to run game.py and see how the play proceeds
