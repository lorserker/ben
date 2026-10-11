## Card play

- **PIMC on Linux x64 works again.** The `libdds.so` that BGADLL (PIMC/PIMCDef)
  calls on Linux was an early DDS 3.0.0 build that accepted only thread index 0.
  BGADLL spreads its solves over indexes 0-7, so about 7 in 8 solves failed and
  were scored as 0 tricks. From 0.8.8.0 to 0.8.8.7, PIMC in the Linux Docker
  images decided on roughly one sample in eight, which showed up as odd cards
  such as underruffing partner instead of discarding. The library is rebuilt from
  DDS 3.1.0 (glibc 2.28 or newer).
- **PIMC on ARM64 can load its solver.** `bin/BGA/linux/arm64/libdds.so` and
  `bin/BGA/windows/arm64/dds.dll` were x86-64 copies, which cannot load next to an
  ARM64 BGADLL. Both are now real ARM64 builds.
- **`pimc_use_discarding` is now a working switch.** The branch it controlled
  never ran: it compared against `"discard"` instead of `"Discard"`, read a
  setting that does not exist, and used a variable before it was set. All shipped
  configs now set it to `True`, which is what BEN has always done, so play is
  unchanged. `False` decides discards with BEN's own double dummy alone.
- **The PIMC/BEN-DD message on a candidate card is complete again.** When the
  card already had a message, the PIMC half was dropped.

## Opening lead

- **The single dummy estimator works** (`double_dummy = False` in `[lead]`, and
  `double_dummy_calculator = False` with `estimator = sde/dde/both` for
  bidding). The model was called with the play network's input shape, the lead
  path fed it 24-card hands where it expects 32 cards, and the bidding path
  crashed on an unset variable.
- **New carding rule `24` against suit contracts** (`lead_from_pips_suit`): top of
  touching honours, 2nd highest from a weak suit, 4th best from an honour, as
  `lead_from_pips_nt = 24` already does against notrump.

## API

- **Systems and carding per request** (the Seat Robot API): `ourSystem`,
  `theirSystem`, `ourCarding` and `theirCarding` choose, for one request only,
  the bidding network and BBA card for each side and the opening-lead carding.
  `GET /` now returns JSON listing the names a request may send and what the
  server plays without them. BEN plays `21GF-GIB`, `21GF` and `SAYC`, and reads
  those plus six robots (`BEN:LIA-21GF`, `BEN:QPLUS-21GF`, `BEN:WBRIDGE5-SAYC`,
  `BEN:BLUECHIP-SAYC`, `BEN:SHARK-SAYC`, `BEN:ROBO-SAYC`). Cardings are the
  system names plus `STANDARD` and `STANDARD-35`. The list is in
  `config/systems.conf`. An unknown name is answered with `400`.
- **`matchtype` no longer leaks between requests.** It changed the server's
  shared setting, so after one matchpoint request every later request without
  `matchtype` was scored as matchpoints. `scoring` is accepted as another name
  for it, and `board` as another name for `dealno`.
- **`details=false` means false.** Any value, including `false`, used to return
  the candidates and samples.
- **Claims are checked against every layout of the hidden cards** (up to 500;
  otherwise 200 random ones). The check used a single random layout, so a claim
  that fails in some layouts was usually accepted.
- **Double dummy runs in parallel in `gameapi`.** gevent turned the solver's
  thread pool into greenlets on one OS thread, so deals were solved one at a
  time. It now uses native threads: an opening lead went from 6.3 s to 1.0 s and
  a bid from 10.3 s to 2.3 s. Found by grand-slam-bridge.
- **Graceful stop in Docker.** As the container's first process `gameapi` ignored
  SIGTERM, so `docker stop` waited out its timeout and then killed it, losing any
  request in progress. Now it stops accepting connections, lets the requests in
  progress finish (at most `BEN_STOP_GRACE_SECONDS`, default 25) and exits: an
  idle stop takes under 2 s instead of the whole timeout. `start_ben_all.sh`
  passes the signal on in the full image. Docker waits only 10 s by default, so
  give it longer (`stop_grace_period: 30s` in compose, or `docker stop -t 30`).
  Suggested by grand-slam-bridge.
- **Removed two settings that never had an effect:** `spawn = 4` and
  `connection_timeout = 120` were set on the server after it was created, which
  gevent ignores. `gameapi` keeps doing what it always did: no limit on
  simultaneous connections and no connection timeout.
- **Faster start and first requests.** `gameapi` loaded every model twice, and
  now loads them once. All 17 models are warmed up before the port opens, so the
  first request no longer pays for TensorFlow tracing.

## Bidding

- **BBA card `GIB-BBO.bbsa`: transfers if RHO doubles** is turned on.
- **New BBA card `Brill.bbsa`.**

## Native libraries

- **BGADLL (PIMC/PIMCDef) updated from 0.9.9.1 to 0.9.9.2** on all five
  platforms. A double dummy solve that fails inside PIMC is now reported once on
  stderr as `BGADLL DDS warning: ...` instead of being silently scored as 0
  tricks, so a broken solver library shows up in the log.
- **SuitC updated from 0.9.0.11 to 0.9.1.0** on Windows, macOS and Linux.
- **The macOS `dds3` extension runs on macOS 11 and newer.** It was built for
  macOS 26.5 only and failed to load on older systems with
  `Symbol not found: __ZNSt3__113__hash_memoryEPKvm`.

## Tools

- **`scripts/check-undefined-names.py`** reports names used but never defined in
  `src/` (needs `pip install pyflakes`). It found the discard bug above.
