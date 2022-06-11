# Training a Neural Network to Bid

This is a tutorial describing how to train a neural network to bid.

What you will need:
- make sure that you have successfully [installed](https://github.com/lorserker/ben/blob/main/README.md#installation) the bridge engine
- the data is contained in the file `bidding_data.zip`
- the script to transform the data into the binary format expected by the neural network is `bidding_binary.py`
- the script which trains the neural network is `bidding_nn.py`

### Instructions

You need to be located in the `scripts/training/bidding` directory when you execute the following steps.

First, activate the environment

```
conda activate ben
```

Unzip the `bidding_data.zip`. You should have a `bidding_data.txt` file now which contains deals and their corresponding auctions like this:

```
K76.965.Q63.9854 AT83.AJ.J9754.73 Q42.84.T2.AKQT62 J95.KQT732.AK8.J
N None P P 1C 1H P 1N P 2N P 3D P 3H P 4H P P P
K83.KQJT.A5.T973 QJT9.A86.Q8.J642 5.973.KJT73.AKQ5 A7642.542.9642.8
E N-S P 1D P 1H P 2C P 2S P 3H P 3N P P P
AJ8.KT875.98.KQ5 Q2.AQJ432.KT64.4 KT9643.9.AQ75.J7 75.6.J32.AT98632
...
```

Run the script to transform the data into binary format. (the first argument is the number of deals in the dataset, the second argument is the file containing the deals. the dataset provided happens to contain 588735 deals)

```
python bidding_binary.py 588735 bidding_data.txt
```

The above command will create two new files: `X.npy` and `y.npy`. `X.npy` contains the inputs to the neural network and `y.npy` contains the expected outputs. Both are stored in numpy array format.

Next, create a new directory where the trained model will be stored in.

```
mkdir model
```

Finally run the trainig script. This will take several hours to complete, but it will save snapshots of the model as it progresses. If you have a GPU, the training will run faster, but not much faster, because GPUs are not so well suited for the type of NN used.

```
python bidding_nn.py
```

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, edit the [code here](https://github.com/lorserker/ben/blob/main/src/nn/models.py#L21) inserting the path to the network which you just trained.
