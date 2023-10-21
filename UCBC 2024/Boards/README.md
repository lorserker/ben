First run
```MergePBNFiles.exe --d .```
to gather all PBN-input i a new file. The new file will be All.PBN

Then run 
```pbn2ben.exe All.pbn```
to generate the file in BEN-format. This file is excluded from GitHub due to size.

Now we need to translate the bidding into binary format, so the next step is:

```binfo_binary.exe input.ben data```

So we are now ready to train the models

```bidding_nn.exe data ..\models```
```binfo_nn.exe data ..\models```

