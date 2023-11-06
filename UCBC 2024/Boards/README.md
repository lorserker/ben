First run
```MergePBNFiles.exe --d .```
to gather all PBN-input i a new file. The new file will be All.PBN

Then run 
```pbn2ben.exe All.pbn```
to generate the file in BEN-format. This file is excluded from GitHub due to size. (Instead of pbn2ben.exe you can use pbn2ben.py from the src directory)

It might be an idea to set the right conventions in BBA, and just rebid all the hands with the right conventions.
Then BEN will be trianed with the right conventions, and not the bidding in the pbn-files. It is still possible to delete boards, that are impossible or just bad boards.

Now we need to translate the bidding into binary format, so the next step is:

```binfo_binary.exe input.ben data```

So we are now ready to train the models

```bidding_nn.exe data ..\models```
```binfo_nn.exe data ..\models```

The number of epcos are important, when training, and from 20 epocs (1 epoc is one full cucle thru the input data) it is acceptable results, but for production like neural nets it is recommended with about 200-500 epocs.
