EPBot 8742, fixes in the EPBot rollout used by the bidding search, a fix for
memory that kept growing while bidding, and a new way to train the bidding
models from EPBot auctions.

## EPBot 8742

- **BEN now ships EPBot 8742** for Windows, Linux and macOS (`bin/BBA`), the
  version of the latest Bridge Bot Analyzer.

## Bidding

- **The EPBot rollout bids every hand at its own seat** (`use_bba_rollout =
  True`). Before, the rollout turned the deal so BEN's hand came first, and
  three things went wrong in that frame:
  - **Fourth seat never called after three passes.** A third-seat pass was
    scored as a pass-out, so its expected score was always 0, and the search
    could pass a hand the network wanted to open.
  - **The vulnerability was swapped** between the sides whenever exactly one
    side was vulnerable.
  - **The opponents bid with BEN's convention card** instead of their own.

  Each call in the rollout now matches the call EPBot makes for that seat on
  its own.
- **Sampling for long auctions** (6 or more bids) no longer accepts weaker
  samples too early: the point where it relaxes the threshold now follows the
  larger number of deals it may draw.

## Memory

- **The memory of a BEN process no longer grows with every new position.**
  TensorFlow's oneDNN kept native memory for every batch size it saw, and the
  number of samples gives a new batch size almost every call. A process could
  grow from 4 GB to 8–9 GB over 1,000 bids. Every model call is now padded to
  one of a small set of batch sizes (at most 25% padding), with the same
  results. 200 bids took 153 s and 3.6 GB, against 179 s and 4.9 GB before.

## Opening leads

- **The single-dummy lead estimator** (`estimator = sde` or `both`) now uses
  the notrump lead model for notrump contracts, and only leads cards the
  leader holds. The shipped configs use `estimator = none` and are not
  affected.

## Monitoring the API server

- **`gameapi` writes a health log**, `logs/gameapi-<host>-<port>-health-<date>.log`:
  the duration and memory of every request, and the memory every 10 minutes.
  A request still running after 120 seconds is reported with its URL, the CPU
  of each thread and the Python stacks, which helps when a server hangs.
  Settings: `BEN_HANG_SECONDS`, `BEN_HANG_REPEAT` and `BEN_HEARTBEAT_MINUTES`.

## Training the bidding models

- **The training boards are bid directly with the EPBot library**, without the
  Bridge Bot Analyzer program, and `Boards/train_bba_model.sh` runs the whole
  process in WSL or Linux: the deals, double-dummy scores, bidding, filtering,
  training of the bidding and bidding-info networks, and a comparison with the
  model in use. `scripts/training/README2.md` describes it.
- **The deal set of October 2026** (1,262,483 deals, including newly generated
  Practice-Bidding-Scenarios deals) and the BEN-21GF boards bid by EPBot 8742
  are on the [training-data release](https://github.com/lorserker/ben/releases/tag/training-data)
  as `Deals-2026-10-22.zip` and `BEN-21GF-8742.zip`.
- **The Moscito bidding and bidding-info models** (`bidding-moscito_2026-09-13`,
  `binfo-moscito_2026-09-13`) are included.

## Requirements

- TensorFlow 2.21 and Keras 3.15, as in `requirements.txt`.
- On Linux, glibc 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9 or newer):
  the EPBot and PIMC libraries need it.
