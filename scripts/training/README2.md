# Train your own robot

Training a robot is quite simple, but to reach a certain level you need boards covering a wide area of different situations.

I have created a set of interesting hands, and if you provide the bidding for that 1 million hands it will be sufficient to create a robot, that bids at least as good as decent club players.

The problem is bidding the hands, but one way is to use the program Bridge Bidding Analyser (BBA)created by Edward Piwowar [https://sites.google.com/view/bbaenglish]

You can find the hands as PBN here [https://www.netbridge.dk/bridge/Upload/BEN/Training/1mill.pbn] - and in BBA-format [https://www.netbridge.dk/bridge/Upload/BEN/Training/1mill.bba]

I have created the bidding for three different setup

- Playing 2/1 as GIB [https://www.netbridge.dk/bridge/Upload/BEN/Training/GIB-Thorvald.ben]
- Play SAYC as planned for the UCBC [https://www.netbridge.dk/bridge/Upload/BEN/Training/BEN-UCBC.ben]
- Playing SAYC against WBridge5 [https://www.netbridge.dk/bridge/Upload/BEN/Training/BEN-WBridge5-SAYC.ben]

Creating the training is done by running a three scripts:

First transform the hands and bidding input binary input using 

```
python bidding_binary.py 
```

Usage: python bidding_binary.py inputfile outputdirectory NS=<x> EW=<y> alternate=True version=2

NS and EW are optional. If set to -1 no information about system is included in the model (Or just not add as parameters).

If NS or EW is set to 0 the hands from that side will not be used for training.

The input file is the BEN-format (1 line with hands, and next line with the bidding).

alternate is signaling, that the input file has both open and closed room, so NS/EW will be alternated. (Default False)

Version is default 2, and can be ignored

So you can download one of the files and execute the script like this

```
python bidding_binary.py BEN-WBridge5-SAYC.ben binary_bin NS=1 EW=99 
```

NS and EW is playing different systems

Now we got the binary representation of the hands and bidding, so we can train the two neural nets used for bidding

```
python bidding_nn.py bidding_bin
```
and
```
python binfo_nn.py ..\bidding\bidding_bin
```
the parameter is slightly different as the two scripts are located in different directories.

The generated models will be place in a subdirectory model

Expect about 24 hours to generate each model

