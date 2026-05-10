# Using gameapi.py from a Bridge Website

This guide describes how a website can drive a bridge game by calling the BEN
REST API exposed by [gameapi.py](src/gameapi.py) running inside a Docker
container.

For the raw endpoint reference, see [README-api.md](README-api.md). This
document focuses on **how a frontend orchestrates a complete deal** — bidding,
opening lead, card play, claim — and the state it must track, since the API
itself is stateless.

---

## 1. Running gameapi.py in Docker

The provided [Dockerfile](Dockerfile) bakes BEN, gameapi.py and a frontend
appserver into one image. Inside the container, [start_ben_all.sh](start_ben_all.sh)
launches gameapi.py with `--host 0.0.0.0` on port **8085**.

### 1.1 Build and run

```bash
docker build -t ben .
docker run -d --name ben \
    -p 8085:8085 \
    ben \
    bash -c "python3 gameapi.py --host 0.0.0.0 --allowed-hosts '*'"
```

Or, if you want the full bundle (gameserver + gameapi + appserver) as set up
in `start_ben_all.sh`:

```bash
docker run -d --name ben -p 8080:8080 -p 8085:8085 -p 4443:4443 ben
```

### 1.2 The `--allowed-hosts` gotcha

gameapi.py validates the HTTP `Host` header. By default only `localhost` and
`127.0.0.1` are accepted; everything else is silently dropped with HTTP **444**.
Inside Docker, the browser will reach the API by container name, by service
name in compose, or by the host machine's address — none of which match the
default allowlist. You will see no logs in the browser, just hung or empty
responses.

Pass the hostname(s) you actually use:

```bash
python3 gameapi.py --host 0.0.0.0 --allowed-hosts 'localhost,127.0.0.1,ben,bridge.example.com'
# or, for development only:
python3 gameapi.py --host 0.0.0.0 --allowed-hosts '*'
```

### 1.3 CORS

The Flask app enables `flask_cors.CORS(app)` with default settings (all
origins, all routes). A browser at `http://localhost:3000` can call
`http://localhost:8085/bid` directly without proxying. In production, put
the API behind your own reverse proxy and tighten this.

### 1.4 Rate limits

Per remote IP: `20000/day`, `5000/hour`, `100/minute`. Localhost and RFC 1918
private ranges (10/8, 192.168/16, 172.16/12 — including Docker's 172.17/16
bridge) are exempted via `is_internal_request()`. A public deployment will
hit limits quickly with one BEN request per played card across many tables —
plan accordingly.

---

## 2. State the website must track

The API is stateless. Each request must carry the full game state. Your
frontend owns:

| State                | Format                                                      |
|----------------------|-------------------------------------------------------------|
| Dealer               | `N` / `E` / `S` / `W`                                       |
| Vulnerability        | `""` / `NS` / `EW` / `Both` (or `All`)                      |
| The four hands       | PBN suit-dotted strings, e.g. `AK97543.K.T3.AK7`            |
| Auction so far       | List of bids (or `ctx` string — see §3)                     |
| Cards played so far  | Concatenated 2-char codes, e.g. `DJDKD3D2` for ♦J ♦K ♦3 ♦2 |
| Whose turn it is     | Derived from dealer + auction length, then trick winner     |
| Tournament scoring   | `mp` (matchpoint) or `imps`                                 |

You only ever send BEN the hand of the seat *currently to act* (plus dummy,
once dummy is exposed). Hidden hands stay on your server.

---

## 3. Encoding cheat-sheet

**Hand (PBN):** `S.H.D.C` ordered from spades to clubs. Use `_` instead of `.`
if you want to avoid URL escaping: `AK97543_K_T3_AK7`. Use `T` for ten.

**Vulnerability (`vul`):** absolute strings — `""` (none), `NS` / `N-S`,
`EW` / `E-W`, `Both` / `All`. (See `parse_vuln()` in
[gameapi.py:495](src/gameapi.py#L495).)

**Auction (`ctx`):** the parser (`parse_ctx_to_bids()`,
[gameapi.py:455](src/gameapi.py#L455)) accepts three styles:

- Dash-separated: `P-1S-P-3N-P-4S-P-P-P`  ← **recommended for new code**
- Concatenated 2-char: `1C--1S--3N------` where `--` = Pass
- Space-separated 2-char: `1C -- 1S -- 3N --`

Bid encodings: `1C`–`7N` as written (use `N`, not `NT`). Pass = `--` or `P` or
`PASS`. Double = `X` or `Db`. Redouble = `XX` or `Rd`.

**Played cards (`played`):** concatenated 2-char codes in **chronological order**
(suit letter + rank): `SJSQSKSA` = ♠J ♠Q ♠K ♠A. Use rank letters
`AKQJT98765432`. If you instead have cards grouped by trick in PBN order
(N→E→S→W per trick), pass `format=true` and the API will reorder.

---

## 4. Driving a complete deal

Let `BASE = "http://localhost:8085"`. The frontend loop has three phases.

### 4.1 Bidding loop

For each seat in turn (starting at dealer), if it's BEN's seat, ask BEN; if
it's a human's seat, take their input. Append to your auction; stop when you
have three consecutive passes after at least one non-pass call (or four
passes if everyone passes).

```js
async function getBid({hand, seat, dealer, vul, auction}) {
  const ctx = auction.join('-');             // dash-separated
  const url = new URL(`${BASE}/bid`);
  url.searchParams.set('hand', hand);
  url.searchParams.set('seat', seat);
  url.searchParams.set('dealer', dealer);
  url.searchParams.set('vul', vul);
  url.searchParams.set('ctx', ctx);
  url.searchParams.set('details', 'true');   // include explanation + alert
  const r = await fetch(url);
  return r.json();   // { bid: "1S", explanation: "...", alert: "False", candidates: [...] }
}
```

The response includes `bid`, `explanation`, `alert` (string `"True"`/`"False"`),
and (with `details=true`) `candidates`, `samples`, `hcp`, `shape`. Always
include `explanation` in the UI for human review of the robot's calls.

For an alert/description of a *human's* bid mid-auction, call:

```
GET /explain?ctx=1N--2C&seat=N&dealer=N&vul=
```

To list every bid the system would consider in a position (useful as a
"hint" panel for the user):

```
GET /bids?ctx=1N--&file_us=GAVIN_ADVANCED&file_them=DEFAULT
```

### 4.2 Opening lead

Once bidding ends, derive declarer and the opening leader (LHO of declarer).
If the leader is BEN:

```js
const r = await fetch(`${BASE}/lead?` + new URLSearchParams({
  hand: leaderHand,
  seat: leaderSeat,
  dealer, vul,
  ctx: auction.join('-'),
  details: 'true'
}));
const { card, candidates, samples } = await r.json();
```

After the lead, the dummy hand is revealed. **Store it now** — every
subsequent `/play` and `/claim` call requires it.

### 4.3 Card play loop

For each of the 51 remaining cards: if the player on lead is BEN, call
`/play` with the full played-card history. The API returns BEN's chosen card,
and your frontend appends it and advances the trick.

```js
async function getPlay({hand, dummy, seat, dealer, vul, auction, played}) {
  const url = new URL(`${BASE}/play`);
  url.searchParams.set('hand', hand);
  url.searchParams.set('dummy', dummy);
  url.searchParams.set('seat', seat);
  url.searchParams.set('dealer', dealer);
  url.searchParams.set('vul', vul);
  url.searchParams.set('ctx', auction.join('-'));
  url.searchParams.set('played', played);    // e.g. "DJDKD3D2S7..."
  url.searchParams.set('details', 'true');
  const r = await fetch(url);
  return r.json();   // { card: "S7", who: "Simulation", player: 3, candidates: [...] }
}
```

Notes:

- `seat` is the seat (N/E/S/W) of the hand whose card you want.
- `hand` is *that* seat's hand — for declarer's plays, send declarer's hand;
  for dummy's plays, send dummy's hand as `hand` and declarer's as `dummy`.
- The response's `player` field is BEN's internal index relative to declarer
  (0=LHO, 1=dummy, 2=RHO, 3=declarer) — useful for sanity-checking but not
  needed by your renderer.
- After each card, your frontend determines the trick winner and the next
  leader; BEN does not maintain that.

### 4.4 Claims

If a player claims N tricks remaining, validate against BEN before accepting:

```
GET /claim?tricks=4&hand=...&dummy=...&seat=S&dealer=N&vul=&ctx=...&played=...
```

Response `result` is a string like `"Contract: 4S Accepted declarers claim
of 10 tricks"` or `"Declarer claimed 11 tricks - rejected 10"`. Decide whether
to accept based on substring or by re-deriving from `tricks`.

### 4.5 One-shot demo: full auto-play

For "watch BEN play itself" demos or test fixtures, `/autoplay` plays a whole
deal start-to-finish:

```
GET /autoplay?deal=N:862.62.AQT52.A96 AQJT9.Q875.97.K7 7543.AT943.8.JT8 K.KJ.KJ643.Q5432&board=5
```

Note the `+` → space conversion (URL-encode the deal string). The response
contains the full auction, contract, declarer, every played card, tricks taken
and score. Useful for warmup checks against a freshly started container — if
this works, the rest will too.

---

## 5. A minimal browser flow

```js
const BASE = 'http://localhost:8085';

const game = {
  dealer: 'N',
  vul: '',
  hands: { N: '...', E: '...', S: '...', W: '...' },   // your server gave these
  benSeats: new Set(['E', 'W']),                        // E and W are robots
  auction: [],
  played: '',
  dummy: null,
  declarer: null,
};

async function callBen(path, params) {
  const url = new URL(`${BASE}${path}`);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

async function maybeBenBid(seat) {
  if (!game.benSeats.has(seat)) return null;
  const r = await callBen('/bid', {
    hand: game.hands[seat],
    seat, dealer: game.dealer, vul: game.vul,
    ctx: game.auction.join('-'),
    details: 'true',
  });
  return r;   // { bid, explanation, alert, ... }
}

async function maybeBenPlay(seat) {
  if (!game.benSeats.has(seat)) return null;
  // If seat is the dummy, declarer's robot decides — send declarer as `seat`
  // with declarer's hand, and dummy's hand as `dummy`.
  const isDummy = seat === dummyOf(game.declarer);
  const actingSeat = isDummy ? game.declarer : seat;
  const r = await callBen('/play', {
    hand: game.hands[actingSeat],
    dummy: game.hands[dummyOf(game.declarer)],
    seat: actingSeat,
    dealer: game.dealer,
    vul: game.vul,
    ctx: game.auction.join('-'),
    played: game.played,
    details: 'true',
  });
  return r;   // { card, candidates, ... }
}

function dummyOf(decl) { return 'NESW'['NESW'.indexOf(decl) + 2 & 3]; }
```

The full game loop (turn dispatcher, trick-winner calculation, score) lives
on the frontend. BEN only answers "given this state, what do I do next?".

---

## 6. Operational checklist

- **Container is up but every call hangs / returns nothing?** Almost always a
  `Host` header rejection. Restart with `--allowed-hosts '*'` to confirm,
  then narrow it down. Look for `Rejected request from host '...'` in
  `/app/logs/gameapi-*.log`.
- **CORS preflight fails from the browser?** `flask_cors.CORS(app)` allows
  all origins — if something blocks, suspect an upstream proxy/CDN, not
  Flask.
- **First request after container start is very slow.** TensorFlow + the
  models load lazily; the first `/bid` or `/play` takes 5–15s. Hit
  `/autoplay` once at deploy time to warm up.
- **`/play` returns "No contract found".** `ctx` was empty or the auction
  hasn't ended in a contract — check that you append the final three passes.
- **`/play` returns "Called as dummy or with wrong dealer / seat".** You sent
  the dummy seat as `seat`. Send declarer's seat with declarer's hand instead;
  BEN treats dummy's plays as decisions made by the declarer.
- **Card encoding mismatches.** `played` uses `T` for ten and is in
  chronological play order, not bridge-table compass order. If you have PBN
  trick-grouped cards instead, pass `format=true`.
- **High-volume deployments.** Each request holds `model_lock_play` /
  `model_lock_bid` — gameapi.py is effectively single-threaded for inference.
  Run multiple containers behind a load balancer rather than relying on one
  container's internal concurrency.
