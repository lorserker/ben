EPBot 8744 with new GIB-BBO and BEN-21GF bidding models trained on its
auctions, Micro Bridge as a system BEN can read with its own models, a faster
declarer PIMC on Linux and macOS, and better claims.

## EPBot 8744 and new bidding models

- **BEN now ships EPBot 8744** for Windows, Linux and macOS (`bin/BBA`).
- **New GIB-BBO, BEN-21GF and Micro Bridge bidding and bidding-info models**,
  trained on the 1,262,483 deals of October 2026 bid by EPBot 8744 with each
  card, the same way as in 0.9.0.2:

  | Card | Models | Identical auctions (new / previous) | IMPs per board vs EPBot (new / previous) |
  |---|---|---|---|
  | GIB-BBO | `GIB-BBO-8744_2026-11-11-E30`, `GIB-BBO-8744-Info_2026-11-11-E30` | 59.3% / 57.6% | 1.79 / 1.86 |
  | BEN-21GF | `BEN-21GF-8744_2026-11-11-E30`, `BEN-21GF-8744-Info_2026-11-11-E30` | 59.4% / 57.7% | 1.86 / 1.91 |
  | Micro Bridge | `MicroBridge-8744_2026-11-12-E30`, `MicroBridge-8744-Info_2026-11-12-E30` | 61.8% / 14.0% (Lia's) | 1.76 / 3.19 |

  Measured in self-play on 20,000 of the training deals against EPBot 8744's
  own auctions. The previous models are the 8742 ones for GIB-BBO and
  BEN-21GF, and Lia's models, which Micro Bridge was read with until now.
  The GIB-BBO and BEN-21GF models are used by every config that used the 8742
  models (`default.conf`, `default_api.conf`, `GIB-BBO*.conf`,
  `BEN-21GF*.conf`, `BEN-UCBC.conf`, `BBA-21GF.conf`, the opponent configs and
  `systems.conf`).

## Systems and convention cards

- **Micro Bridge** can be given as `theirSystem` (`BEN:MBRIDGE`) to the API.
  BEN reads it with the new Micro Bridge models; it cannot play it.
  The card is renamed `BBA/CC/MicroBridge.bbsa` (was `Micro Bridge 13.4.bbsa`).
- **The GIB-BBO and Brill cards** use the 5NT king ask as inviting
  (`King ask by 5NT inviting = 1`). The new GIB-BBO models are trained with it.
- **New WJ (Polish Club) and Meckwell (Precision) cards** replace the old ones
  in `BBA/CC`.
- **JEC**: a new config `Jec.conf` with bidding and bidding-info models
  (`bidding-JEC_2025-10-22-E30`, `binfo-JEC_2025-10-22-E30`) and the
  convention card `BBA/CC/JEC.bbsa` (GIB-BBO with JEC's own agreements).
- **Pidgin**: the `bidding-pidgin_2026-11-04` and `binfo-pidgin_2026-11-04`
  models with `Robots/pidgin.conf` and `opponent/pidgin.conf`.
- **The GIB-BBO opponent config** (`opponent/GIB-BBO.conf`) now names GIB-BBO
  and reads the opponents with the GIB-BBO card. It said GIB 21GF and used
  `GIB-21GF.bbsa` before, so an opponent loaded with `--opponent GIB-BBO` was
  only partly read as GIB-BBO.

## Card play

- **Declarer PIMC respects `pimc_wait`.** It waited until every queued sample
  was solved, and on Linux and macOS the Haglund DDS backend also solved the
  same position once for every legal card. On a small Linux server a `/play`
  request could take 150 s. With the two reported deals on a 2-CPU server,
  declarer PIMC went from 71 s to about 3 s per request, and the whole request
  from 44-99 s to 9-10 s, with the same cards played.
- **Claims use the suits an opponent has shown out of.** The claim check now
  knows where the missing cards in such a suit are, as double dummy does, so
  a claim that depends on a known finesse is no longer rejected.

## Requirements

- TensorFlow 2.21 and Keras 3.15, as in `requirements.txt`.
- On Linux, glibc 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9 or newer).
