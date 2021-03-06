
## Bidding data generation with [Edward Piwowar's Bridge Bidding Analyzer](https://sites.google.com/view/bbaenglish/bridge-bidding-analyser)

This describes how to use an adaptation of [this example](https://sites.google.com/view/bbaenglish/bridge-bidding-analyser/for-programmers/sample-code-multiple-bots) to generate a lot of bidding data.

The source code is written in Visual Basic, and is contained in the file [Module1.vb](Module1.vb)

This has to be compiled in Visual Studio using the DLL `EPBot64.dll`

Let's say that we compile it to `Bidder.exe`, then in can be used as in the following command

```
Bidder.exe < epbot_input_sample.txt > epbot_output_sample.txt
```

After that the epbot output is further transformed into the format which is needed for the [bidding neural network training](../../training/bidding)

```
python epbot_to_bidding_data.py < epbot_output_sample.txt > bidding_data_sample.txt
```
