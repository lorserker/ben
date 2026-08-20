# Training a Neural Network to Bid

This is a tutorial describing how to train a neural network to bid. There is a [bridgewinners article](https://bridgewinners.com/article/view/bridge-ai-how-neural-networks-learn-to-bid/) to go with this tutorial.

What you will need:
- make sure that you have successfully [installed](https://github.com/lorserker/ben/blob/main/README.md#installation) the bridge engine
- the data is contained in the file `bidding_data.zip`
- the script to transform the data into the binary format expected by the neural network is `bidding_binary_keras.py`
- the script which trains the neural network is `bidding_nn_keras.py`

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

Run the script to transform the data into binary format. (the first argument is the number of deals in the dataset, the second argument is the file containing the deals and the third is the script output. the dataset provided happens to contain 588735 deals)

```
mkdir -p binary/bidding models/bidding

python bidding_binary_keras.py bidding_data.txt None binary/bidding NS=1 EW=1
```
Here there are 2 optional parameters as you can specify system used for both NS and EW.
Specifying -1 for both will create a neural network without any information about bidding system.

0 = 2/1 (GIB)
1 = Sayc
2 =

The above command will create two new files into the `binary/bidding` folder: `x.npy` and `y.npy`. `x.npy` contains the inputs to the neural network and `y.npy` contains the expected outputs. Both are stored in numpy array format.

Then, run the trainig script. This will take several hours to complete, but it will save snapshots of the model as it progresses. If you have a GPU, the training will run faster, but not much faster, because GPUs are not so well suited for the type of NN used.

```
python bidding_nn_keras.py binary/bidding bidding
```

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, set the path to the network which you just trained in a configuration file - either `src/config/default.conf` or a new configuration file of your own.

#### How to continue training an already trained model

This part describes how you can load an already trained model and continue training it (without the training starting from scratch)

Let's say your already trained model is stored in the `model` folder and you want to continue training it and then store the results to the `model2` folder. You can do this by running the [bidding_nn_continue.py](bidding/bidding_nn_continue.py) script.

```
mkdir -p models/bidding-bis

python bidding_nn_continue.py models/bidding/bidding-1000000 models/bidding-bis
```

### Training a bidding-info model

This model is used to estimate the strength and shape of hidden hands based on their bidding (i.e meaning of bids)

It is needed if you want to use a bidder neural network in the engine (so it can get information from the bidding)

There is no separate binary step any more - `bidding_binary_keras.py` already writes `HCP.npy` and `SHAPE.npy` alongside `x.npy`, `y.npy` and `z.npy`, and the bidding-info network is trained directly from that same directory.

From `scripts/training/bidding_info` (run under WSL):

```
python binfo_nn_keras.py ../bidding/binary/bidding binfo
```

The first argument is the directory produced by `bidding_binary_keras.py`, the second is the name given to the generated model.

### Making a test run

To test the neural network, it is possible to feed it a file of test hands and see how it bids them. `testrun.py` takes a configuration file and an input file of deals; sample input files are in `scripts/training/data/`.

```
python bidding/testrun.py default.conf ../data/1deal.txt
```

this will generate the auctions as they are bid by the model and writes them in this format:

```
S E-W J3.AKQJT9.42.AK3 85.875.K53.QT965 Q9642.6.JT98.J84 AKT7.432.AQ76.72 P-1D-X-P-1S-P-2H-P-2S-P-3H-P-3S-P-P-P
S N-S 8.AT732.JT8.AJ63 AKT92.J4.752.T84 7653.Q986.AQ9.Q9 QJ4.K5.K643.K752 P-1C-1H-1S-2C-2H-3H-3S-P-P-P
E None AT.K4.QJ9542.Q53 96542.765.8.T972 K3.AQ98.AKT73.AK QJ87.JT32.6.J864 P-2C-P-3D-P-4H-P-4S-P-4N-P-5H-P-7D-P-P-P
```

This can be used for debugging, to see if the NN bids test hands as expected


### More data

More data is available for download. It was generated with [Edward Piwowar's Bidding Analyzer](https://sites.google.com/view/bbaenglish) for different systems (1 million deals each)

- [SAYC](https://bridgedatasets.s3.eu-west-1.amazonaws.com/epbot/sayc_bidding_data.txt.gz)
- [2/1](https://bridgedatasets.s3.eu-west-1.amazonaws.com/epbot/21gf_bidding_data.txt.gz)
- [Polish Club](https://bridgedatasets.s3.eu-west-1.amazonaws.com/epbot/wj_bidding_data.txt.gz)
- [Precision](https://bridgedatasets.s3.eu-west-1.amazonaws.com/epbot/pc_bidding_data.txt.gz)
