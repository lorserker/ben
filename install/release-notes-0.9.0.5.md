Fixes the memory leak in the double dummy solver that 0.9.0.4 brought
(#206), and makes `gameapi` stricter about the requests it accepts and easier
to run behind a reverse proxy.

## Memory leak in bridge-solver fixed (#206)

- **A long-running `gameapi` no longer grows with every deal.** In 0.9.0.4
  bridge-solver kept each thread's search caches for good: a thread that
  exited leaked them, and `FreeMemory()` did not reach them. `gameapi` grew
  by about 0.25 GB per deal until it ran out of memory.
- Now a thread's caches are freed when the thread exits, and `FreeMemory()`
  frees the caches of every thread. The libraries are built from
  bridge-solver `dad3495`
  ([BSalita/bridge-solver#1](https://github.com/BSalita/bridge-solver/pull/1)),
  and they search exactly as fast as before.
- Measured with `gameapi` in Docker (Linux x64), playing deals card by card
  through `/lead` and `/play`: with 0.9.0.4's library memory went from
  2.4 GB to 13.3 GB in 24 deals. With the new one it levels off at about
  3.3 GB and stays there (46 deals). On macOS (M1) the memory the library
  holds stays at 1.6 MB however many threads come and go; before it grew
  about 56 MB for every 8 threads.
- Replaced for Windows x64, Linux x64, Linux arm64 and macOS arm64, both
  BEN's own copy (`bin/bsolver`) and the `dds` library that PIMC uses
  (`bin/BGA`, `bin/dds.dll`, `bin/libdds.so`, `bin/darwin`).

## gameapi

From Bob Salita (#203):

- **Malformed requests are refused with 400** and an error text, instead of
  being answered with a guess: a hand or dummy that is not 13 different
  cards, a missing hand, seat, dealer or dummy, `/play` asked of dummy's
  seat, the wrong seat on `/lead`, or an unknown `vul`. A 12-card hand on
  `/lead` used to stop the whole server.
- **`/bid` returns `alert` as a JSON boolean** (`true`/`false`). It was the
  text `"True"`/`"False"`, and `"False"` counts as true in JavaScript.
  **Clients that compared it with the text must be changed.**
- `/play` for a passed-out board with nothing played answers `boardOver`,
  as it does when all 52 cards have been played.
- `GET /` reports the engine, the version and, in a Docker image, the build.

## Behind a reverse proxy

From Bob Salita (#204):

- The web pages and the API pages work when BEN is served through a reverse
  proxy or tunnel (the game servers at `/ws/<port>/`, the API at
  `/gameapi/<port>/`). Opened directly they work as before.
- **`gameapi --trusted-proxies`** names the proxies whose `X-Forwarded-For`
  and `CF-Connecting-IP` headers are believed. By default these headers are
  now ignored: anyone could send them and get round the rate limits.
  `README-api.md` describes the setup.
- **`appserver.py --direct-ports`** is for a proxy that routes by port: the
  pages then keep `wss://host:444N/` and `https://host:8085`. Without it the
  pages behave as described above.

## Training

From Bob Salita (#205):

- The bidding networks train with the data on the GPU: 30 epochs of both
  networks on 3.8 million examples take about 16 minutes instead of 5 hours
  on an RTX 5080, giving equivalent models.
- The conversion of training data to binary runs in worker processes, about
  13 times faster, with byte-identical output.
- `eval_bidding_nn.py` is five times faster.
