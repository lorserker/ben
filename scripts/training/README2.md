# Training a bidding model on EPBot auctions

BEN's bidding network learns from deals together with their auctions. The auctions come from EPBot, the bidding engine of the Bridge Bidding Analyser (BBA) by Edward Piwowar: <https://sites.google.com/view/bbaenglish>. Every board is bid with one convention card at all four seats, and the network is trained on those auctions.

EPBot is called directly through the native library BEN already ships in `bin/BBA` (the same one BEN consults while it plays), so the BBA program itself is not needed.

The tools come from three places:

- **In BEN:** `Boards/bid_with_epbot.py`, `bidding_binary_keras.py`, `bidding_nn_keras.py` and `binfo_nn_keras.py`.
- **In [Bridge-Robot-Utilities](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities):** `MergePBNFiles`, `Split_PBN`, `ExtractDatumScore` and `ComputeDatumScores`. The executables are attached to its [releases](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities/releases). `ComputeDatumScores` is source only.
- **Deals for special bidding situations:** [Practice-Bidding-Scenarios](https://github.com/ThorvaldAagaard/Practice-Bidding-Scenarios).

If you only want to train, skip to [Step 6](#step-6---convert-the-boards-to-binary). The finished `-OK_boards.pbn` files are published on the [training-data release](https://github.com/lorserker/ben/releases/tag/training-data), one zip per system (`GIB-BBO-8730-OK_boards.zip`, `BEN-21GF-8730-OK_boards.zip` and so on). One million raw deals in BBA format are also there: [1mill.bba](https://github.com/lorserker/ben/releases/download/training-data/1mill.bba).

## Unattended run

`Boards/train_bba_model.sh` runs the whole process below in WSL (or on Linux), from the deals to evaluated models:

```
cd /mnt/d/GitHub/ben
bash Boards/train_bba_model.sh BEN-21GF                # deals already in Boards/
bash Boards/train_bba_model.sh BEN-21GF --scenarios    # first generate new Practice-Bidding-Scenarios deals
```

It merges the deals in the subfolders of `Boards/`, adds par scores for new deals to the datum file, bids every deal with EPBot, splits against par, converts all the OK boards to binary, trains the bidding and the bidding-info network side by side, and measures how well the new model (and the model the card's config uses now) reproduces the auctions on 20,000 of the training boards. The merged deals are written once each to `BBA/Boards/Deals-<date>.pbn`; another system bids the same deals with `--deals BBA/Boards/Deals-<date>.pbn`. The run is named after the card and the EPBot version in `bin/BBA` (e.g. `BEN-21GF-8740`); logs and intermediate files go to `BBA/Run-<name>/`. With the April 2025 deals it takes about 30 minutes before training and 2 hours of training.

Options: `--pull` (with `--scenarios`: `git pull` Practice-Bidding-Scenarios first), `--deals FILE` (bid this PBN instead of merging `Boards/`), `--datum FILE`, `--test N`, `--epochs N`, `--from STEP` (resume at `deals`, `datum`, `bid`, `split`, `binary`, `train` or `eval`), `--to STEP` (stop after it) and `--no-train` (`--to binary`). Run it without arguments for the full help.

Prerequisites:

- **A Python environment in WSL** with TensorFlow 2.21 / Keras 3.15 (as BEN), `endplay`, `colorama`, `psutil` and `gputil`, set with `PY` (default `~/anaconda3/envs/ben-k315/bin/python`).
- **`DatumScores.pkl` with double-dummy tables** as `BBA/Boards/DatumScores-DD.pkl` (`DatumScores.zip` from the [latest Bridge-Robot-Utilities release](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities/releases/latest)). New deals are added to it automatically, which takes a while for many new deals.
- **[Bridge-Robot-Utilities](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities)** checked out next to BEN (or set `UTILS`). Its `Split_PBN.py` must take the file name on the command line (after 1.0.22).
- **For `--scenarios`:** [Practice-Bidding-Scenarios](https://github.com/ThorvaldAagaard/Practice-Bidding-Scenarios) next to BEN (or `PBS`), and `dealerv2` (default `/usr/local/games/DealerV2_4/Prod/dealerv2`, or `DEALERV2`).
- **Memory and disk:** about 30 GB of WSL memory while both networks train, and about 15 GB of disk for the binary training data.

The steps are described one by one below.

## What the training data contains

- **One card at all four seats.** A BEN-21GF file is BEN-21GF against BEN-21GF. There is one file per system.
- **Every auction is bid by EPBot.** The deal sources, including the boards tracked from BEN, contribute only dealer, vulnerability and the cards, so no human call reaches the training auctions.
- **The result is double-dummy.** Each contract is scored with the double-dummy number of tricks. (Boards from the BBA program, such as the 8730 files, were scored with BBA's single-dummy play instead.)
- **`-OK_boards` is a quality filter, not only "has a datum".** The datum is the double-dummy par score (`OptimumScore`, kept in `DatumScores.pkl`). `Split_PBN` keeps a board only if it:
  - is not a third or later copy of the same deal,
  - has a par score,
  - is not a doubled contract that made, and
  - scored within 2000 points of par.

  Doubled contracts that went down stay in, unless they are more than 2000 from par. Badly bid boards are therefore removed only when they are disasters; ordinary bidding mistakes stay in the data.

## Naming

Everything from one run carries the convention card and the EPBot version, `<CC>-<version>`:

| What | Name |
| --- | --- |
| Convention card | `BBA/CC/BEN-21GF.bbsa` |
| Bid boards | `BBA/Boards/BEN-21GF-8740.pbn` |
| After Split_PBN | `BBA/Boards/BEN-21GF-8740.pbn-OK_boards.pbn` and the other split files |
| Binary training data | `scripts/training/bidding/BEN-21GF-8740/` |
| Bidding model | `BEN-21GF-8740_<date>-E<epoch>.keras` |
| Bidding-info model | `BEN-21GF-8740-Info_<date>-E<epoch>.keras` |

The script prints the EPBot version it used, and writes it in the header of the PBN file.

## Producing the boards

The commands are Windows commands, run from a prompt with the `ben` environment active, from the BEN folder unless stated otherwise. `BBA/Boards` and `BBA/Run*/` are ignored by git, so the large files can live there.

### Step 1 - collect the deals

1. **Practice-Bidding-Scenarios (optional):** `train_bba_model.sh --scenarios` generates 500 deals for every scenario script in the repository's `dlr/` folder with `dealerv2`. It leaves out each script's dealer and actions, so the dealer rotates from board to board, and seeds by the scenario name, so a run is reproducible. Every file is written a second time with the hands turned one seat (`-rotated`), and the result replaces the files in `Boards/Practice Bidding Scenarios`. New deals need par scores in the datum file (step 3), which takes a while.
2. **Deals tracked from BEN:** BEN-format lines are collected in `Boards/BEN/deals.ben`. From `Boards/BEN`, run the commands in `run.cmd`:

   ```
   python clean_and_sort.py > deals_cleaned.ben
   python ben2pbn.py deals_cleaned.ben > deals.pbn
   ```

   They sort the lines, remove duplicates, and write each deal as a PBN board with dealer, vulnerability and cards only.

### Step 2 - merge all PBN files

`MergePBNFiles` merges every PBN file in the folder tree, **including an old `All.PBN` and any stray copies**, so either delete those first or merge into a separate folder:

```
cd Boards
MergePBNFiles.exe -d . -o ..\BBA\Run<n>\All.PBN -r
python Boards\dedup_deals.py BBA\Run<n>\All.PBN BBA\Boards\Deals-<date>.pbn
```

`dedup_deals.py` keeps each deal once (same cards, dealer and vulnerability) with only Board, Dealer, Vulnerable and Deal. The deals file is the input for every system; the merged `All.PBN` can be deleted. The October 2026 set, `Deals-2026-10-22.pbn`, has 1,262,483 deals.

### Step 3 - bid the boards with EPBot

```
python Boards\bid_with_epbot.py BBA\Boards\Deals-<date>.pbn BEN-21GF BBA\Boards\BEN-21GF-8742.pbn --datum DatumScores.pkl
```

- The second argument is the card name in `BBA/CC`.
- The deals are bid in parallel on all cores. 1.17M boards take about 3 minutes.
- Repeated boards (same cards, dealer and vulnerability) are bid once.
- `--datum` must be a `DatumScores.pkl` **with double-dummy tables**. Download `DatumScores.zip` from the [latest Bridge-Robot-Utilities release](https://github.com/ThorvaldAagaard/Bridge-Robot-Utilities/releases/latest). It covers all 1.26M deals of the October 2026 set, including the Practice-Bidding-Scenarios deals. The `DatumScores.pkl` in the training-data release has par scores but no tables.
- `--limit 2000` bids only the first 2000 deals, for a quick test.

Boards whose deal is not in the pickle are written without a result and end up in `-missing-DD.pbn` in step 4. Add them with `ComputeDatumScores` from Bridge-Robot-Utilities and run step 3 again:

```
python src\ComputeDatumScores.py missing.pbn DatumScores.pkl
```

Each board also gets `[OptimumScore]`, `[ParContract]` and the full double-dummy `[OptimumResultTable]`, which `eval_bidding_nn.py` uses in step 9.

Alerts are written as `=n=` with the meaning in a `[Note]`, as BBA does. Training ignores them unless `alert_supported=True` is given in step 6.

### Step 4 - filter against the par score

```
Split_PBN.exe BBA\Boards\BEN-21GF-8740.pbn DatumScores.pkl
```

The pickle argument is optional; without it `Split_PBN` uses `DatumScores.pkl` in the current folder. Started without arguments, it opens a file dialog. (Versions up to 1.0.22 only have the dialog.) 1.17M boards take about 15 minutes.

| File | Contents |
| --- | --- |
| `<name>.pbn-OK_boards.pbn` | the boards that passed the filter - **this is the file training uses** |
| `<name>.pbn-db_making.pbn` | doubled contracts that made |
| `<name>.pbn-db_not_making.pbn` | doubled contracts that went down (these are also in OK_boards unless more than 2000 from par) |
| `<name>.pbn-disaster.pbn` | more than 2000 points from par |
| `<name>.pbn-duplicates.pbn` | the third and later copies of a deal. Two copies are kept, because Practice-Bidding-Scenarios writes every deal a second time rotated, with another dealer |
| `<name>.pbn-missing-DD.pbn` | no par score |

### Step 5 (optional) - compare with an earlier run

To see how much the new EPBot version changes the auctions, compare the auctions board by board with the previous file of the same card. Between the BBA 8730 run and EPBot 8740, 45% of the auctions were identical and the opening call changed on 3.2% of the boards.

## Training

### Step 6 - convert the boards to binary

From `scripts/training/bidding`:

```
python bidding_binary_keras.py ..\..\..\BBA\Boards\BEN-21GF-8740.pbn-OK_boards.pbn None BEN-21GF-8740 NS=1 EW=1 rotate=True n_cards=24 max_occurrences=50
```

The arguments are:

- **input file:** BEN or PBN format. The second argument is a second input file, or `None`.
- **output directory:** created if it does not exist.
- **`NS` / `EW`:** which sides are used for training. `-1` includes no system information in the model; `0` leaves that side's hands out.
- **`rotate=True`:** rotates every deal so North is first to bid.
- **`n_cards`:** the number of cards the network sees per hand. The default is 24, with the small cards merged.
- **`max_occurrences`:** caps the number of boards with the same auction, so the common auctions do not swamp the rare ones. The default is 25.
- **`alternate=True`:** use this when the file has an open and a closed room, so NS and EW alternate.
- **`alert_supported=True`:** also records alerts.

1.16M boards give 3.9M training examples in about 10 minutes; `x.npy` is 12 GB. The output is `x.npy`, `y.npy`, `z.npy`, `HCP.npy` and `SHAPE.npy`, and both networks below are trained from it.

### Where to train

TensorFlow on Windows has no GPU support, so train in WSL (Ubuntu) on the GPU, in a conda environment with the same TensorFlow and Keras versions as BEN (see [requirements.txt](../../requirements.txt): TensorFlow 2.21, Keras 3.15). The training scripts also need `psutil` and `gputil`:

```
pip install psutil gputil
```

Loading the binaries takes about as much memory as `x.npy` (12 GB), so give WSL enough memory (`.wslconfig`).

### Step 7 - train the bidding network

From `scripts/training/bidding`:

```
python bidding_nn_keras.py BEN-21GF-8740 BEN-21GF-8740 lstm_size=256 decay_steps=3000
```

The first argument is the directory from step 6 and the second is the name of the model. The options and their defaults are:

- `lstm_size=128`
- `n_layers=3`
- `batch_size=512`
- `epochs=30`
- `learning_rate=0.001`
- `decay_steps=1000`: the learning rate is multiplied by 0.96 every `decay_steps` batches.

The latest models were trained with `lstm_size=256 decay_steps=3000`. 256 units bid clearly better than 128, at about 13% more bidding time inside BEN. A larger `decay_steps` keeps the learning rate from decaying to nearly zero before training ends.

### Step 8 - train the bidding-info network

From `scripts/training/bidding_info`:

```
python binfo_nn_keras.py ../bidding/BEN-21GF-8740 BEN-21GF-8740-Info
```

The path is relative because the two scripts are in different directories. The info network predicts each player's HCP and shape from the auction, and BEN uses it to sample hands. This script takes no options; it always trains 30 epochs.

Both networks can be trained at the same time on one GPU. On an RTX 4090 with the 1.16M BEN-21GF boards, an epoch takes about 3 minutes for the bidding network and 3.5 minutes for the info network, so 30 epochs take about 1.5 and 2 hours. The models are written to a `model` subdirectory of each script's folder, one file per epoch.

### Step 9 - test the models

Every OK board is used for training; none is held out. The goal is that the model reproduces the auctions it was trained on, so measure that on a sample of the training boards (the script takes 20,000 spread evenly over the file), and run the model the config uses now on the same boards for comparison:

```
python eval_bidding_nn.py model\BEN-21GF-8742_<date>-E30.keras sample.pbn --ns 1 --ew 1 --n_cards 24
```

It reports how often the network makes the same call as EPBot, the auctions it reaches in self-play, and their double-dummy cost (from `[OptimumResultTable]`). The script was written for the Brill model, so its labels say "Brill" for the reference auctions. `--dump diffs.txt` writes the differences out for inspection.

To use the models in BEN, copy them to `models/TF2models/` and point the `[bidding]` section of a config at them:

```
[bidding]
bidder = models/TF2models/BEN-21GF-8740_<date>-E30.keras
info = models/TF2models/BEN-21GF-8740-Info_<date>-E30.keras
```

Then play a match against the previous model to see whether it bids better.
