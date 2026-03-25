# BEN API Documentation

BEN (Bridge Engine Neural) provides a REST API for bridge bidding, card play, and analysis. This document describes all available endpoints.

## Starting the Server

```bash
python gameapi.py
```

The API uses `default_api.conf` as configuration (override with `--config`).
Default port is 8085 (override with `--port`).

### Host Validation

The API validates the `Host` header on incoming requests. Only requests from allowed
hosts are accepted — all others are silently rejected with HTTP 444.

By default, only `localhost` and `127.0.0.1` are allowed. When running in Docker or
behind a reverse proxy, you must add your hostname(s):

```bash
# Allow specific hosts (comma-separated)
python gameapi.py --allowed-hosts 'localhost,127.0.0.1,ben'

# Allow all hosts (useful in Docker / trusted networks)
python gameapi.py --allowed-hosts '*'
```

> **Tip:** If you get HTTP 444 "No Response" on every request, this is almost certainly
> the cause. Check the logs for a "Rejected request from host" warning.

## Common Parameters

These parameters are used across multiple endpoints:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `hand` | Hand in PBN format (dots or underscores as separators) | `AK97543.K.T3.AK7` or `AK97543_K_T3_AK7` |
| `seat` | Player position | `N`, `E`, `S`, `W` |
| `dealer` | Dealer position | `N`, `E`, `S`, `W` |
| `vul` | Vulnerability: `@V` = We vul, `@V` = They vul | `@v@V` (both vul), empty (none) |
| `ctx` | Bidding context as concatenated 2-char bids | `1C--1S` = 1C-Pass-1S |
| `dummy` | Dummy's hand in PBN format | `QJ87.A95.K63.T42` |
| `played` | Cards played as concatenated 2-char cards | `SJSQSKSA` |
| `tournament` | Scoring type | `mp` (matchpoint) or `imps` |
| `details` | Include extended information | `true` |

### Bid Encoding in `ctx`

| Bid | Encoding |
|-----|----------|
| Pass | `--` or `Pa` |
| Double | `Db` |
| Redouble | `Rd` |
| 1C-7N | As-is: `1C`, `2H`, `3N`, etc. |

---

## Endpoints

### `/bid` - Get Bid Recommendation

Returns the recommended bid for a given hand and auction state.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `hand` | Yes | Your hand in PBN format |
| `seat` | Yes | Your position (N/E/S/W) |
| `dealer` | Yes | Dealer position (N/E/S/W) |
| `vul` | Yes | Vulnerability string |
| `ctx` | Yes | Current auction (can be empty for opening) |
| `details` | No | Include candidates/samples if `true` |
| `tournament` | No | `mp` for matchpoint scoring |

**Example Request:**
```
GET /bid?hand=AK97543.K.T3.AK7&seat=S&dealer=N&vul=&ctx=----&tournament=mp
```

**Response:**
```json
{
  "bid": "1S",
  "who": "NN",
  "quality": "Good",
  "candidates": [
    {"call": "1S", "insta_score": 0.998}
  ],
  "hcp": [9.2, 7.3, 6.5],
  "shape": [1.9, 4.0, 3.7, 3.3, ...]
}
```

---

### `/bids` - List All Possible Bids

Returns all possible bids for an auction context with their meanings from the bidding system.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `ctx` | Yes | Current auction |
| `file_us` | No | Convention card file for our side |
| `file_them` | No | Convention card file for opponents |

**Example Request:**
```
GET /bids?ctx=1N--
```

**Response:**
```json
[
  {"bid": "2C", "description": "Stayman"},
  {"bid": "2D", "description": "Transfer to hearts"},
  {"bid": "2H", "description": "Transfer to spades"},
  ...
]
```

---

### `/lead` - Get Opening Lead

Returns the recommended opening lead after the auction.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `hand` | Yes | Your hand in PBN format |
| `seat` | Yes | Your position (N/E/S/W) |
| `dealer` | Yes | Dealer position |
| `vul` | Yes | Vulnerability string |
| `ctx` | Yes | Complete auction |
| `details` | No | Include candidates/samples if `true` |

**Example Request:**
```
GET /lead?hand=KJ53.KJ7.AT92.K5&seat=W&dealer=N&vul=@v&ctx=1N--3N------
```

**Response:**
```json
{
  "card": "CA",
  "who": "Simulation (MP)",
  "quality": "Good",
  "hcp": [7.7, 2.5, 12.4],
  "shape": [0.8, 4.3, 4.2, 3.7, ...],
  "candidates": [
    {"card": "CA", "insta_score": 0.325, "expected_tricks_sd": 5.28, "p_make_contract": 1.0, "expected_score_sd": 236},
    {"card": "CK", "insta_score": 0.276, "expected_tricks_sd": 5.28, "p_make_contract": 1.0, "expected_score_sd": 236}
  ],
  "samples": [...]
}
```

---

### `/play` - Get Card to Play

Returns the recommended card to play during the card play phase.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `hand` | Yes | Your hand in PBN format |
| `dummy` | Yes | Dummy's hand in PBN format |
| `seat` | Yes | Your position (N/E/S/W) |
| `dealer` | Yes | Dealer position |
| `vul` | Yes | Vulnerability string |
| `ctx` | Yes | Complete auction |
| `played` | Yes | Cards played so far (concatenated) |
| `details` | No | Include candidates/samples if `true` |
| `format` | No | `true` if cards are in PBN trick format |

**Example Request:**
```
GET /play?hand=AK97543.K.T3.AK7&dummy=QJ87.A95.K63.T42&seat=S&dealer=N&vul=&ctx=1S--4S------&played=DJDKD3D2
```

**Response:**
```json
{
  "card": "S7",
  "who": "Simulation",
  "quality": "Good",
  "player": 3,
  "matchpoint": false,
  "candidates": [...],
  "samples": [...]
}
```

**Note:** The `player` field indicates the position relative to declarer:
- `0` = LHO (Left Hand Opponent)
- `1` = Dummy
- `2` = RHO (Right Hand Opponent)
- `3` = Declarer

---

### `/claim` - Verify a Claim

Verifies whether a claim of a certain number of tricks is valid.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `tricks` | Yes | Number of tricks claimed |
| `hand` | Yes | Your hand in PBN format |
| `dummy` | Yes | Dummy's hand in PBN format |
| `seat` | Yes | Your position (N/E/S/W) |
| `dealer` | Yes | Dealer position |
| `vul` | Yes | Vulnerability string |
| `ctx` | Yes | Complete auction |
| `played` | Yes | Cards played so far |

**Example Request:**
```
GET /claim?tricks=10&hand=AK97.K.T3.AK7&dummy=QJ87.A95.K63.T42&seat=S&dealer=N&vul=&ctx=4S------&played=...
```

**Response:**
```json
{
  "tricks": 10,
  "result": "Contract: 4S Accepted declarers claim of 10 tricks"
}
```

---

### `/autoplay` - Auto-Play Complete Board

Plays a complete board with BEN handling all 4 positions for bidding and card play.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `deal` | Yes | PBN deal string (4 hands space-separated) |
| `board` | No | Board number 1-16 for dealer/vul (default: 1) |
| `dealer` | No | Override dealer (N/E/S/W) |
| `vul` | No | Override vulnerability (None/NS/EW/Both) |

**Example Request:**
```
GET /autoplay?deal=862.62.AQT52.A96 AQJT9.Q875.97.K7 7543.AT943.8.JT8 K.KJ.KJ643.Q5432&board=5
```

**Response:**
```json
{
  "deal": "862.62.AQT52.A96 AQJT9.Q875.97.K7 7543.AT943.8.JT8 K.KJ.KJ643.Q5432",
  "dealer": "N",
  "vulnerability": "N-S",
  "auction": ["1D", "1S", "2H", "PASS", "4H", "PASS", "PASS", "PASS"],
  "contract": "4H",
  "declarer": "S",
  "tricks": 10,
  "score": 420,
  "ns_score": 420,
  "play": ["S2", "SA", "S3", "SK", ...],
  "elapsed": 45.23
}
```

---

### `/contract` - Predict Contract

Predicts the likely contract given two hands (for partnership simulation).

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `hand` | Yes | First hand in PBN format |
| `dummy` | Yes | Second hand in PBN format |
| `seat` | Yes | Position of first hand |
| `vul` | Yes | Vulnerability string |

**Response:**
```json
{
  "3N": {
    "score": 0.85,
    "Tricks": [9, 10],
    "Percentage": [0.45, 0.35]
  },
  "4H": {
    "score": 0.12,
    "Tricks": [10],
    "Percentage": [0.80]
  }
}
```

---

### `/explain` - Explain a Bid

Returns explanation for a specific bid in context.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `ctx` | Yes | Auction up to and including the bid to explain |

**Example Request:**
```
GET /explain?ctx=1N--2C
```

---

### `/explain_auction` - Explain Full Auction

Returns explanations for all bids in an auction.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `ctx` | Yes | Complete auction to explain |

---

## Response Fields

### Common Response Fields

| Field | Description |
|-------|-------------|
| `bid` / `card` | The recommended action |
| `who` | Algorithm that made the decision (NN, Simulation, etc.) |
| `quality` | Confidence level (Good, Fair, etc.) |
| `candidates` | Alternative options with scores |
| `samples` | Sample hands generated for analysis |
| `hcp` | HCP distribution estimates |
| `shape` | Shape distribution estimates |

### Candidate Fields

| Field | Description |
|-------|-------------|
| `call` / `card` | The bid or card |
| `insta_score` | Neural network score |
| `expected_tricks_sd` | Expected tricks from simulation |
| `p_make_contract` | Probability of making contract |
| `expected_score_sd` | Expected score from simulation |

---

## Error Handling

All endpoints return HTTP 400 with an error message on failure:

```json
{
  "error": "An error occurred: Invalid hand format"
}
```

Rate limiting is enforced (default: 10000/hour, 100/minute). When exceeded:

```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "limit": "100 per 1 minute"
}
```

---

## Usage Examples

### Complete Bidding Sequence

```python
import requests

BASE = "http://localhost:8085"

# North opens
r = requests.get(f"{BASE}/bid", params={
    "hand": "AKQ.KQJ.AT98.KQ2",
    "seat": "N", "dealer": "N", "vul": "", "ctx": ""
})
print(r.json()["bid"])  # "2C" (strong)

# East passes
r = requests.get(f"{BASE}/bid", params={
    "hand": "432.432.432.4322",
    "seat": "E", "dealer": "N", "vul": "", "ctx": "2C"
})
print(r.json()["bid"])  # "PASS"
```

### Getting Opening Lead

```python
r = requests.get(f"{BASE}/lead", params={
    "hand": "KJ53.KJ7.AT92.K5",
    "seat": "W",
    "dealer": "N",
    "vul": "@v",
    "ctx": "2C--2D--2N--3N------"
})
print(r.json()["card"])  # e.g., "DA"
```

---

## Integration with Other Systems

The BEN API is designed to be stateless - you must provide all game state information with each request. This makes it easy to integrate with:

- Online bridge platforms (as a robot player)
- Training tools (hint systems)
- Analysis tools (post-game review)
- Tournament directors (claim verification)

For integration with the Brill bidding system, use the same endpoint format - Brill's `/bid` endpoint accepts identical parameters.
