## Card play

- **ACE now uses both signals the engine produces.** Ace exposes a win rate
  (probability of making) and a utility (average overtrick margin). BEN read only
  the win rate, which tied every making card at 100% and threw away the overtrick
  preference Ace had already computed - so it could play the wrong card from a set
  of equally "winning" ones. Card results are now emitted as
  `(remaining_tricks, score, p_make)` including overtricks, matching PIMC's
  convention so carding and claim thresholds stay calibrated.
- **Updated Ace engine** (`Ace.dll`), which renamed the evaluation field from
  `Value` to `Winrate`. Adds the `BEN-21GF-ACE1` configuration alongside
  `BEN-21GF-ACE`.
- **Defender no longer spins on a closed socket** while waiting for a card (#185).

## Bidding

- **The bid explainer no longer receives the Hint and Alert controls**, which
  raised a `KeyError` (#184).

## Servers

- **Fixed the daily log rollover in gameapi.** `doRollover` never advanced
  `rolloverAt`, so from the first midnight onward *every* log record closed and
  reopened the log file - several times per request. Long-running API servers got
  steadily slower.
- **Removed a forced `gc.collect()` on every API request.** It ran a full
  generation-2 collection per request, at a cost that grows with the resident heap.
- **New pre-flight checks (`src/preflight.py`).** Verifies the interpreter, the
  `dds3` extension and PIMC's DDS backend before a server starts. A mismatched
  Python now produces one clear message instead of a chained traceback, and a PIMC
  backend that would abort partway through the first trick is caught before the
  server accepts a connection. `runservers.sh` runs it and warns when it falls back
  off the virtualenv.
- **New `src/run_api_bbo.sh`** - starts the Play API on `GIB-BBO.conf` with the
  repo virtualenv, and refuses to start on an unsupported interpreter rather than
  silently falling back to the system `python3`.
- **The web UI server dropdown defaults to the port a single instance listens on**
  (#186).

## Tooling

- **DDS benchmarking.** Every DDS call can be recorded with its input and output
  (`BEN_DDS_RECORD`) and replayed later with `src/ddsolver/ddsreplay.py`. Model
  timing is now broken down per purpose.
- **Release automation.** `install/Release.cmd` builds the four packages and
  publishes the GitHub release with their zips attached.

## Known issues

- On macOS, PIMC's DDS backend fails to load under hardened-runtime Python builds
  (the python.org installer): the server starts and bids normally, then aborts
  during the first trick PIMC is asked to play. Use a Homebrew or conda Python -
  the documented virtualenv setup does - or export `DYLD_LIBRARY_PATH` to
  `bin/BGA/macos/<arch>`. Pre-flight now detects and explains this. See #187.
