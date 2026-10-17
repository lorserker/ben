## Requirements

- **TensorFlow 2.21 and Keras 3.15 are now required** (were 2.18 and 3.5).
  The new play and lead models are saved with Keras 3.15 and do not load in
  Keras 3.5, so update the environment together with this release
  (`pip install -r requirements.txt`). Older models still load in the new
  versions and, with the changes below, give the same output; the 2024 lead
  models need the `-v2` files, which have their inputs in an explicit order.
- **BEN checks the versions when it starts.** `gameapi.py`, `game.py`,
  `gameserver.py` and `table_manager_client.py` stop with an explanation when
  TensorFlow or Keras is too old, instead of failing while loading a model with
  *could not be deserialized properly*:

  ```
  This BEN needs TensorFlow 2.21 or newer and Keras 3.15 or newer,
  but this Python has TensorFlow 2.18.1 and Keras 3.5.0.
  ...
  Update the packages from the BEN folder:
      pip install -r requirements.txt
  ```

  Set `BEN_SKIP_VERSION_CHECK=1` to start anyway, for example with a
  configuration that uses only older models. A TensorFlow or Keras newer than
  the tested versions (2.21.0 and 3.15.1) only gives a warning. The `.exe`
  packages include the right versions.
- **Two Keras changes that silently broke BEN are handled:**
  - Keras 3.6 and older passed the lead models' two inputs in the wrong order
    when they were given as a dict, and the 2024 lead models were trained that
    way. Keras 3.7 fixed the order, which made those models' leads close to
    random (52.9% → 9.6% on the training leads). The lead models are now called
    with a list, and new lead models are trained the same way.
  - `TimeDistributed` in Keras 3.15 returns one output step per board instead of
    one per bid when the number of steps is not known in advance, which is how
    BEN calls the bidding and bidding-info models. `nn/keras_compat.py` works
    around it when those models are loaded.
- **The frozen builds include the DLL the new `ml_dtypes` needs.** Without it
  every `.exe` stopped on `import tensorflow` with *DLL load failed while
  importing _ml_dtypes_ext*.

## Card play

- **New play models: a hybrid of dense layers and LSTM.** The 2024 play models
  saw each trick on its own; the LSTM the original models had was lost when
  they were moved to Keras. The new models (two dense layers of 512, then two
  LSTM layers of 256) see the whole play so far. On 64,000 held-out deals per
  model they predict the card actually played better at all eight seats, e.g.
  declarer in suit contracts 73.6% → 76.7%.
- **Measured in matches on the same 500 deals:**

  | Cards chosen by | Result for 0.9.0.0 |
  |---|---|
  | PIMC and double dummy (the shipped settings) | +33 IMPs, not significant |
  | Double dummy over the samples (PIMC off) | +16 IMPs, not significant |
  | The play models alone (`play_nn_only`) | **+735 IMPs, +1.47 per board** |

  As a card player on its own the new model is much stronger. With double
  dummy deciding the cards, the two versions are equal within what 1000 boards
  can show.
- **Card play is about twice as fast.** In the matches above 0.9.0.0 used 36–37%
  less thinking time in total and 38–50% less per card, from the changes under
  Speed.
- **New setting `play_nn_only`** (`[cardplay]`, default `False`): every card is
  the play model's own choice, with no double dummy, PIMC or SuitC. For
  measuring the play models.

## Opening lead

- **New lead models (`LeadV3-Suit-Both`, `LeadV3-NT-Both`)**, trained on the
  expert leads in the WBC data and the robot leads in the Jack data together.
  Their input is now encoded exactly as BEN encodes it at runtime: the stored
  training data had the bidding features from an older bidding-info model, up
  to 2 HCP off what BEN computes. Results on held-out deals:

  | | Suit | Notrump |
  |---|---|---|
  | BEN's complete lead, 2024 models (DDOLAR) | 87.2% | 73.2% |
  | BEN's complete lead, new models (DDOLAR) | 87.0% | 74.5% |
  | The WBC expert's lead on the same deals | 86.2% | 70.0% |

  BEN leads at expert level with either model. The new models are a little
  better against notrump and predict a robot opponent's lead much better, which
  is what BEN uses the lead model for when it checks its samples during the
  play.

## Speed

None of these change a decision: with the same seed every board plays the same.

- **DDS 3.1.0.** The double-dummy solver extension (`bin/dds3-*`) is rebuilt
  from the DDS v3.1.0 release on Windows, Linux and macOS. Replaying 2740
  recorded DDS calls from a real session gives the same answer for every call,
  in 22% less time (187 s → 146 s on 32 threads). The Linux build now runs on
  any distribution with glibc 2.25 or newer (it needed 2.38).

- **The play models' predictions are shared** between the three checks of the
  samples in one decision, instead of being computed three times.
- **Sample hands are converted between the 32-card and 24-card formats in one
  step** instead of one hand at a time through text (about 100× faster).
- **Hidden cards are shuffled with `Generator.permuted`**, which draws the same
  random numbers as the per-sample loop it replaces.
- **Filtering samples no longer copies all of them twice.**

## Game API

- **`/play` and `/claim` reject an impossible deal** with a 400 error that names
  the card, e.g. *Not a possible deal: S7 is in both hand and dummy*. Before, the
  deal reached the double-dummy solver, which refused it, and the request failed
  with *'NoneType' object has no attribute 'items'*. The `/play` example in
  README-api.md was such a deal and is replaced by a real one.
- **Spot cards given as `x` are filled in correctly.** The highest spot an `x`
  can stand for (the 7, or the 9 when the hand shows no 8 or 9) was never chosen,
  an `x` could repeat a spot card named in the same suit, and a suit with more
  `x`s than free spot cards made the request hang.

## Models

- All shipped configurations use the new play and lead models. The 2024 models
  stay in `models/TF2models`.
- **`models/onnx`**: the eight play models and the two lead models are
  regenerated from the new models (checked against the Keras models on real
  deals: the same card every time).

## Tools

- **`scripts/training/playing/play_nn_keras.py`** trains the hybrid models:
  dense layers in front of the LSTM (`dense_layers`, `dense_units`), a held-out
  validation set and early stopping, and data fed from memory (an epoch on 1.2
  million deals takes about 20 s on a GPU). **`eval_play_nn.py`** compares models
  on the held-out deals.
- **`scripts/training/opening lead/gen_lead_data.py`** builds lead training data
  with BEN's runtime encoding; **`lead_train.py`** and **`lead_eval.py`** train
  and compare lead models (coverage, DDOLAR, and how well the actual lead is
  predicted). **`fix_lead_input_order.py`** re-saves a 2024 lead model with its
  inputs in an explicit order.
