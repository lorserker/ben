## Bidding

- **Fall back to the neural network when the simulation cannot be trusted.** With
  very few matching samples the MP values collapse to exactly -1/0/+1, so the
  estimate is noise. The sample-count guard also now catches the case where fewer
  than the minimum were found, not only exactly the minimum.

## Scoring

- **`factor_to_translate_to_mp` goes from 10 to 500** across all configurations.
  `calculate_mp_score` has returned -1..+1 since 0.8.8.2 rather than 0..100, so the
  old factor was 50x too small.

## Packaging

- **SuitC now works in frozen builds.** A PyInstaller build has no `python.exe` and
  no `suitc_worker.py` on disk, so SuitC could not spawn its worker. It now
  re-launches the BEN executable itself with a worker sentinel, handled at each
  entry point before argparse and the heavy imports.

## Servers

- **`run_api_bbo.sh` defaults to port 80**, matching what `runservers.sh` exposes as
  `api-bbo-80` and what the reverse proxy forwards to. Use `PORT=8085` for an
  unprivileged local test.

## Training

- **Training data is now published as a GitHub release** (tag `training-data`)
  instead of a personal web server: the board sets for all ten systems, the play
  data, and `DatumScores.pkl`. New `install/Compress-Boards.ps1` and
  `install/Publish-TrainingData.ps1` build and upload it.
- **The training documentation has been brought back in line with the code.**
  Sixteen scripts named in the docs no longer existed under those names after the
  Keras migration.
- **Fixed play-model training on Linux and WSL.** `play_nn_keras.py` loaded
  `x_`/`y_*.npy` while `binary_keras_suit_nt.py` writes `X_`/`Y_*.npy` - harmless on
  Windows, fatal on a case-sensitive filesystem.
