A bug-fix release: two card-play errors, and opening leads and play against
another robot that were decided on far too few samples.

## Card play

- **Declarer no longer leads into a sure ruff instead of drawing the last
  trump.** With only claim-making cards left, every adjustment was set to zero,
  including the reward for drawing trumps. Drawing the last trump then only tied
  with a side-suit card, and the tie could go to the side suit. On board 59 of a
  match against WBridge5, declarer in 4♥ led a club with one trump out, the
  defence ruffed, and 11 cold tricks became 10. The trump reward is now kept for
  trump cards.
- **An ace led from a long suit no longer misleads the samples**
  (`lead_convention = True`). The check of the opening lead against the lead
  convention accepted an ace lead against a suit contract only from A, Ax or
  Axx. When the leader had led the ace from Axxx or longer, every sample where
  they held that length was down-weighted, so declarer believed partner of the
  leader still held the suit. In the board above that was what made the club
  look safe. The ace is now a valid lead from any length against a suit
  contract; the NT rules are unchanged.

## Playing against another robot

- **The opponents' bidding is read with their own bidding-info network.** The
  opponent configs (`config/opponent/*.conf`) turned `use_biddinginfo` off,
  because BEN's own bidding-info network reads the opponents' calls as BEN's
  system. Without it the hidden hands were dealt at random, and only about one
  deal in a thousand fits a constructive auction like 1NT – 2♣ – 2♠ – 4♠. An
  opening lead was then chosen on about 20 samples instead of 200, which in one
  match made BEN lead the ♣10 from A10985 and let 4♠ make.
- **New setting `opponent_info`** (`[bidding]`, in the opponent configs): the
  opponents' bidding-info network. When sampling, the HCP and shape of the two
  opponents come from it and partner's from BEN's own network. This applies to
  the samples for bidding, the opening lead and card play. Without the setting
  nothing changes.
- **The opponent configs name the info network that belongs to their bidding
  model and turn `use_biddinginfo` on again.** In the example above the lead
  has about 250 samples again, in the same time. `GIB.conf` has no
  `opponent_info`, because its bidding model has no info network; it uses
  BEN's own for all hands.
- **`systems.conf` gives the robot systems their info network**, and a request
  with `theirSystem` uses it for the opponents in the same way.

## Configuration

If you use your own copy of an opponent config, add `opponent_info` and set
`use_biddinginfo = True` as in the shipped files, for example:

```
[bidding]
opponent = models/TF2models/WBridge5-Sayc-8730_2025-04-20-E30.keras
opponent_info = models/TF2models/WBridge5-Sayc-Info-8730_2025-04-20-E30.keras

[cardplay]
use_biddinginfo = True
```

## Measured

Each fix was checked on the positions it is for (the 4♥ and 4♠ boards above).
In a 96-board match against WBridge5 the result was within the noise of a
single match (−15 IMPs before, −13 after): these situations come up a few
times in a hundred boards.
