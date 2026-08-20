# Train your own robot

Training a robot is quite simple, but to reach a certain level you need boards covering a wide area of different situations.

The training input is deals together with their bidding. The hard part is producing the bidding, and the way it is done here is with the program Bridge Bidding Analyser (BBA) created by Edward Piwowar: <https://sites.google.com/view/bbaenglish>

## Where the training input lives

All training input is kept under `BBA/Boards/` in the working tree. It is **not** in git - it is several tens of GB of PBN files and is excluded in `.gitignore`.

Files are named `<System>-<BBAversion>.pbn`, for example `GIB-BBO-8730.pbn`, and each source file is split into several derived files:

| File | Contents |
| --- | --- |
| `<name>.pbn-OK_boards.pbn` | the boards that matched a datum score - **this is the file training uses** |
| `<name>.pbn-db_making.pbn` | boards where the contract is making |
| `<name>.pbn-db_not_making.pbn` | boards where the contract is going down |
| `<name>.pbn-disaster.pbn` | boards with a very bad result |
| `<name>.pbn-duplicates.pbn` | boards removed as duplicates |
| `<name>.pbn-missing-DD.pbn` | boards still missing a double dummy result |

If you do not want to generate your own boards, a public starting set of one million deals is available in BBA-format: <https://www.netbridge.dk/bridge/Upload/BEN/Training/1mill.bba>

## Producing the boards

This is the pipeline used to create the files in `BBA/Boards/`. The `MergePBNFiles`, `Split_PBN` and
`ExtractDatumScore` tools come from [Bridge-Robot-Utilities](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities);
`pbn2bba` and `pbn2ben` are in BEN itself under `src/`.


1. Check whether there are new or updated files in `Practice_Bidding_Scenarios`, sync the repository, then run `run.sh` from `py/custom` under WSL and copy the generated PBN files into the BEN repository.
2. For boards tracked from BEN, run `clean_and_sort.py` and `ben2pbn.py`.
3. Gather all the PBN files, delete any existing `all.pbn`, and merge with `MergePBNFiles.exe -d . -r`.
4. Convert to BBA format with `pbn2bba.exe All.PBN`.
5. Split into bulks of 200.000 boards with `Boards/split_and_testfordublicates.py`.
6. Copy the files and `run.cmd` to `BBA` and let every system bid the boards - `runBBA.cmd GIB-BBO`, `runBBA.cmd Lia-21GF` and so on. Every opponent and system must bid the boards.
7. Collect the results into `BBA/Boards/<System>-<version>.pbn`.
8. Run `Split_PBN` to separate good and bad boards - this identifies the boards where the datum score is missing.
9. Generate double dummy scores for the boards in the `missing-DD` file (Bridge Composer, Tools -> Double Dummy all boards), then run `ExtractDatumScore` to update `DatumScores.pkl`.
10. Run the split again so the boards can be matched against the datum score. The result is the `-OK_boards.pbn` file used for training.

## Step 1 - convert the boards to binary

From `scripts/training/bidding`:

```
python bidding_binary_keras.py inputfile inputfile2 outputdirectory NS=<x> EW=<y> alternate=True rotate=True n_cards=<n> alert_supported=True max_occurrences=<z>
```

- the input file is in BEN-format or PBN-format; use `None` for `inputfile2` if there is only one input file
- `outputdirectory` is the system name, and the binary files are written to a directory of that name
- `NS` and `EW` select which side is used for training. If set to -1 no information about system is included in the model. If set to 0 the hands from that side are not used
- `alternate` signals that the input file has both open and closed room, so NS/EW are alternated (default False)
- `rotate` rotates all deals so North is first to bid
- `n_cards` is the number of cards in the deck (default 24)
- `alert_supported` records alerts in the output
- `max_occurrences` limits the number of boards per auction (default 25)

The invocations actually used are kept in `scripts/training/bidding/run.cmd`, for example:

```
python bidding_binary_keras.py ..\..\..\..\BBA\Boards\GIB-BBO-8730.pbn-OK_boards.pbn None GIB-BBO NS=1 EW=1 rotate=True n_cards=24 max_occurrences=25
```

This writes `x.npy`, `y.npy`, `z.npy`, `HCP.npy` and `SHAPE.npy` into `scripts/training/bidding/GIB-BBO/`. The same binary output feeds both neural networks below.

## Step 2 - train the bidding network

From `scripts/training/bidding`:

```
python bidding_nn_keras.py inputdirectory system
```

where `inputdirectory` is the directory produced in step 1 and `system` is the name given to the generated model, for example:

```
python bidding_nn_keras.py GIB-BBO GIB-BBO-8730
```

## Step 3 - train the bidding info network

From `scripts/training/bidding_info`, run under WSL:

```
python binfo_nn_keras.py ../bidding/GIB-BBO GIB-BBOInfo-8730
```

The path is relative because the two scripts are in different directories. The invocations used are in `scripts/training/bidding_info/run.cmd`.

The generated models are placed in a `model` subdirectory. Expect about 24 hours to generate each model.
