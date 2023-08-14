## Generating data with GIB

This describes how to make GIB bid and play out a deal and save the data how it bid and played.

We use the version of GIB which comes with BBO, so we have to [install the BBO windows client](https://www.bridgebase.com/intro/installation_guide_for_bbo.php) first.

There will be a program `bridge.exe` installed, which is actually GIB, and we can feed it hands to play.

```
bridge.exe a < input.gib > output.txt
```

The file [input.gib](input.gib) contains the hands we want the bot to play (it's convenient to put several deals into the file so it can play for a while)

The file [output.txt](output.txt) will be created by the program and contains data about the bidding and about the play.

When the play has finished execute `parse_gib_output.py` to merge [input.gib] and [output.txt] into [gib.txt]

```
python parse_gib_output.py > gib.txt
```

And finally use `gib_to_training.py` to create the file [training.txt] used for training


```
python gib_to_training.py <gib.txt >training.txt
```
