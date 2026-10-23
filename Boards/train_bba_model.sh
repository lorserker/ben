#!/usr/bin/env bash
# Train a BEN bidding model on auctions bid by EPBot, unattended. Runs in WSL (or Linux).
#
# Usage: Boards/train_bba_model.sh CC [options]
#   CC                 convention card in BBA/CC, e.g. BEN-21GF
#   --scenarios        first generate new deals with Practice-Bidding-Scenarios
#                      and copy them into "Boards/Practice Bidding Scenarios"
#   --pull             with --scenarios: git pull Practice-Bidding-Scenarios first
#   --deals FILE       bid this deals file (e.g. BBA/Boards/Deals-<date>.pbn) instead of
#                      merging the deals in Boards/
#   --datum FILE       DatumScores.pkl with double-dummy tables
#                      (default BBA/Boards/DatumScores-DD.pkl)
#   --test N           hold out the last N OK boards from training (default 0: train on all)
#   --epochs N         epochs for the bidding network (default 30)
#   --from STEP        start at STEP: deals, datum, bid, split, binary, train, eval
#   --to STEP          stop after STEP
#   --no-train         stop after the binary training data (--to binary)
#
# Environment (defaults in brackets):
#   PY        Python with TensorFlow, endplay and colorama  [~/anaconda3/envs/ben-k315/bin/python]
#   UTILS     Bridge-Robot-Utilities checkout               [<BEN>/../Bridge-Robot-Utilities]
#   PBS       Practice-Bidding-Scenarios checkout           [<BEN>/../Practice-Bidding-Scenarios]
#   DEALERV2  dealerv2 executable                           [/usr/local/games/DealerV2_4/Prod/dealerv2]
#
# The run is named <CC>-<EPBot version>; the version is read from bin/BBA/linux/x64/libEPBot.so.
# Merging writes the deals once each to BBA/Boards/Deals-<date>.pbn, for other systems to bid too.
# Logs and intermediate files go to BBA/Run-<name>/ (ignored by git).

set -euo pipefail

BEN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/anaconda3/envs/ben-k315/bin/python}"
UTILS="${UTILS:-$BEN/../Bridge-Robot-Utilities}"
PBS="${PBS:-$BEN/../Practice-Bidding-Scenarios}"
DEALERV2="${DEALERV2:-/usr/local/games/DealerV2_4/Prod/dealerv2}"

usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[ $# -ge 1 ] || usage
CC="$1"; shift
SCENARIOS=0; PULL=0; DEALS=""; DATUM="$BEN/BBA/Boards/DatumScores-DD.pkl"
TEST=0; EPOCHS=30; FROM=deals; TO=eval
while [ $# -gt 0 ]; do
    case "$1" in
        --scenarios) SCENARIOS=1 ;;
        --pull) PULL=1 ;;
        --deals) DEALS="$(realpath "$2")"; shift ;;
        --datum) DATUM="$(realpath "$2")"; shift ;;
        --test) TEST="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --from) FROM="$2"; shift ;;
        --to) TO="$2"; shift ;;
        --no-train) TO=binary ;;
        *) usage ;;
    esac
    shift
done

STEPS=(deals datum bid split binary train eval)
step_index() { local i; for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && echo "$i" && return; done; echo "Unknown step $1" >&2; exit 1; }
FROM_I=$(step_index "$FROM")
TO_I=$(step_index "$TO")
run_step() { local i; i=$(step_index "$1"); [ "$i" -ge "$FROM_I" ] && [ "$i" -le "$TO_I" ]; }
log() { echo "$(date '+%H:%M:%S') $*"; }

# Check the prerequisites before anything long starts
[ -x "$PY" ] || { echo "Python not found: $PY (set PY)"; exit 1; }
[ -f "$BEN/BBA/CC/$CC.bbsa" ] || { echo "Convention card not found: $BEN/BBA/CC/$CC.bbsa"; exit 1; }
[ -f "$DATUM" ] || { echo "Datum file not found: $DATUM (DatumScores.zip from the latest Bridge-Robot-Utilities release)"; exit 1; }
[ -f "$UTILS/src/Split_PBN.py" ] || { echo "Bridge-Robot-Utilities not found: $UTILS (set UTILS)"; exit 1; }
"$PY" -c "import tensorflow, endplay, colorama, psutil, GPUtil" 2>/dev/null \
    || { echo "$PY needs tensorflow, endplay, colorama, psutil and gputil"; exit 1; }

VERSION=$(cd "$BEN/src" && "$PY" -c "
from bba.BBA import BBABotBid, EPBotNative
BBABotBid.get_dll()
print(EPBotNative().version())")
NAME="$CC-$VERSION"
RUN="$BEN/BBA/Run-$NAME"
BOARDS="$BEN/BBA/Boards/$NAME.pbn"
mkdir -p "$RUN"
log "Training $NAME (EPBot $VERSION), work folder $RUN"

# 1. Deals: optional new scenarios, the deals tracked from BEN, then merge everything in Boards/*/
if run_step deals; then
    if [ "$SCENARIOS" = 1 ]; then
        [ -x "$DEALERV2" ] || { echo "dealerv2 not found: $DEALERV2 (set DEALERV2)"; exit 1; }
        if [ "$PULL" = 1 ]; then
            log "Pulling Practice-Bidding-Scenarios"
            git -C "$PBS" pull --ff-only
        fi
        log "Generating Practice-Bidding-Scenarios deals (log: $RUN/scenarios.log)"
        SC="$RUN/scenarios"
        rm -rf "$SC"; mkdir -p "$SC/dlr" "$SC/pbn"
        # Every scenario script without its dealer, generate/produce and actions (and comment blocks):
        # 500 deals, the dealer rotating from board to board, seeded by the scenario name
        "$PY" - "$PBS/dlr" "$SC/dlr" > "$SC/commands.txt" <<'EOF'
import sys, os, re, hashlib
src, dst = sys.argv[1], sys.argv[2]
for name in sorted(os.listdir(src)):
    if not name.endswith('.dlr'):
        continue
    text = open(os.path.join(src, name), encoding='utf-8', errors='replace').read()
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    out = ['generate 400000000', 'produce 500']
    for line in text.split('\n'):
        word = line.strip().split(' ')[0]
        if word == 'action':
            break
        if word not in ('dealer', 'generate', 'produce'):
            out.append(line)
    out.append('action printpbn')
    with open(os.path.join(dst, name), 'w') as f:
        f.write('\n'.join(out) + '\n')
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big') % (2**32 - 1)
    print(f"{os.path.splitext(name)[0]} {seed}")
EOF
        export DEALERV2 SC
        xargs -P "$(nproc)" -L 1 sh -c '"$DEALERV2" "$SC/dlr/$0.dlr" -s "$1" > "$SC/pbn/$0.pbn" 2>> "$SC/errors.log" || echo "Failed: $0" >> "$SC/errors.log"' \
            < "$SC/commands.txt" > "$RUN/scenarios.log" 2>&1
        # Without the dates, and every file once more with the hands turned one seat
        "$PY" - "$SC/pbn" <<'EOF'
import sys, os
folder = sys.argv[1]
turn = {'N:': 'E:', 'E:': 'S:', 'S:': 'W:', 'W:': 'N:'}
for name in sorted(os.listdir(folder)):
    path = os.path.join(folder, name)
    lines = [l for l in open(path).read().split('\n') if not l.startswith('[Date ')]
    if not any(l.startswith('[Deal ') for l in lines):
        print(f"No deals in {name}")
        os.remove(path)
        continue
    open(path, 'w').write('\n'.join(lines))
    rotated = [l[:7] + turn.get(l[7:9], l[7:9]) + l[9:] if l.startswith('[Deal ') else l for l in lines]
    open(os.path.join(folder, name[:-4] + '-rotated.pbn'), 'w').write('\n'.join(rotated))
EOF
        grep -h "Failed" "$SC/errors.log" 2>/dev/null || true
        rm -f "$BEN/Boards/Practice Bidding Scenarios/"*.pbn
        cp "$SC/pbn/"*.pbn "$BEN/Boards/Practice Bidding Scenarios/"
        log "$(ls "$SC/pbn/"*.pbn | wc -l) scenario files in Boards/Practice Bidding Scenarios"
    fi

    if [ "$BEN/Boards/BEN/deals.ben" -nt "$BEN/Boards/BEN/deals.pbn" ]; then
        log "Converting the deals tracked from BEN"
        (cd "$BEN/Boards/BEN" && "$PY" clean_and_sort.py deals.ben > deals_cleaned.ben \
            && "$PY" ben2pbn.py deals_cleaned.ben > deals.pbn)
    fi

    if [ -z "$DEALS" ]; then
        # Only the subfolders: Boards/ itself holds the old All.PBN and stray copies
        log "Merging the deals in Boards/*/"
        rm -rf "$RUN/sources"; mkdir -p "$RUN/sources"
        for d in "$BEN/Boards"/*/; do cp -r "$d" "$RUN/sources/"; done
        "$PY" "$UTILS/src/MergePBNFiles.py" -d "$RUN/sources" -o "$RUN/All.PBN" -r > "$RUN/merge.log" 2>&1
        rm -rf "$RUN/sources"
        # Each deal once (same cards, dealer and vulnerability), kept for other systems to bid
        DEALS="$BEN/BBA/Boards/Deals-$(date +%Y-%m-%d).pbn"
        (cd "$BEN" && "$PY" Boards/dedup_deals.py "$RUN/All.PBN" "$DEALS")
        rm -f "$RUN/All.PBN"
    fi
    echo "$DEALS" > "$RUN/deals.txt"
    log "$(grep -c '^\[Deal ' "$DEALS") deals in $DEALS"
fi
# The deals file of this run, also when resuming with --from
if [ -z "$DEALS" ]; then
    [ -f "$RUN/deals.txt" ] || { echo "No deals file for this run: give --deals or start from the deals step"; exit 1; }
    DEALS=$(cat "$RUN/deals.txt")
fi

# 2. Par scores and double-dummy tables for deals not yet in the datum file (only new deals take time)
if run_step datum; then
    log "Adding missing deals to $DATUM (log: $RUN/datum.log)"
    "$PY" "$UTILS/src/ComputeDatumScores.py" "$DEALS" "$DATUM" > "$RUN/datum.log" 2>&1
    tail -1 "$RUN/datum.log"
fi

# 3. Bid every deal with EPBot
if run_step bid; then
    log "Bidding with EPBot $VERSION"
    "$PY" "$BEN/Boards/bid_with_epbot.py" "$DEALS" "$CC" "$BOARDS" --datum "$DATUM" | tee "$RUN/bid.log" | tail -2
fi

# 4. Filter against the par score
if run_step split; then
    log "Splitting $BOARDS"
    (cd "$(dirname "$BOARDS")" && "$PY" "$UTILS/src/Split_PBN.py" "$BOARDS" "$DATUM" > "$RUN/split.log" 2>&1)
    grep "generated" "$RUN/split.log"
    if [ -f "$BOARDS-missing-DD.pbn" ]; then
        echo "Some deals have no par score: $BOARDS-missing-DD.pbn"; exit 1
    fi
fi

# 5. Convert the OK boards to binary training data. All of them are trained on (unless --test),
#    and an evenly spread sample of 20,000 is kept to measure how well the model reproduces them
if run_step binary; then
    "$PY" - "$BOARDS-OK_boards.pbn" "$RUN/train.pbn" "$RUN/test.pbn" "$RUN/sample.pbn" "$TEST" <<'EOF'
import sys, os
src, train, test, sample, n_test = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
if not n_test and os.path.exists(test):
    os.remove(test)
text = open(src, encoding='utf-8').read()
boards = ['\n'.join(line for line in b.split('\n') if not line.startswith('%'))
          for b in text.split('\n\n') if '[Deal ' in b]
cut = max(0, len(boards) - n_test)
step = max(1, cut // 20000)
parts = [(train, boards[:cut]), (sample, boards[:cut:step][:20000])]
if n_test:
    parts.append((test, boards[cut:]))
for path, part in parts:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('% PBN 2.1\n% EXPORT\n\n' + '\n\n'.join(b.strip('\n') for b in part) + '\n\n')
print(f"{cut} training boards" + (f", {len(boards) - cut} held out" if n_test else ""))
EOF
    log "Converting to binary (log: $RUN/binary.log)"
    (cd "$BEN/scripts/training/bidding" && "$PY" bidding_binary_keras.py "$RUN/train.pbn" None "$NAME" \
        NS=1 EW=1 rotate=True n_cards=24 max_occurrences=50 > "$RUN/binary.log" 2>&1)
    grep "created" "$RUN/binary.log" | tail -1
fi

# 6. Train the bidding and the bidding-info network side by side on the GPU
if run_step train; then
    log "Training $NAME and $NAME-Info (logs: $RUN/train_bidding.log, $RUN/train_info.log)"
    (cd "$BEN/scripts/training/bidding" && "$PY" bidding_nn_keras.py "$NAME" "$NAME" \
        lstm_size=256 decay_steps=3000 epochs="$EPOCHS" > "$RUN/train_bidding.log" 2>&1) &
    BIDDING_PID=$!
    (cd "$BEN/scripts/training/bidding_info" && "$PY" binfo_nn_keras.py "../bidding/$NAME" "$NAME-Info" \
        > "$RUN/train_info.log" 2>&1) &
    INFO_PID=$!
    FAILED=0
    wait $BIDDING_PID || { echo "Bidding network failed, see $RUN/train_bidding.log"; FAILED=1; }
    wait $INFO_PID || { echo "Bidding-info network failed, see $RUN/train_info.log"; FAILED=1; }
    [ "$FAILED" = 0 ] || exit 1
fi

# 7. Measure how well the new model, and the model the card's config uses now, reproduce the auctions
if run_step eval; then
    MODEL=$(ls -t "$BEN/scripts/training/bidding/model/${NAME}_"*.keras | head -1)
    INFO=$(ls -t "$BEN/scripts/training/bidding_info/model/${NAME}-Info_"*.keras | head -1)
    CURRENT=$(sed -n 's/^bidder *= *//p' "$BEN/src/config/$CC.conf" 2>/dev/null | tr -d '\r' | head -1)
    : > "$RUN/eval.txt"
    for set in sample test; do
        [ -f "$RUN/$set.pbn" ] || continue
        for m in "$MODEL" ${CURRENT:+"$BEN/$CURRENT"}; do
            log "Evaluating $(basename "$m") on $RUN/$set.pbn"
            echo "=== $(basename "$m") on $set.pbn" >> "$RUN/eval.txt"
            (cd "$BEN/scripts/training/bidding" && "$PY" eval_bidding_nn.py "$m" "$RUN/$set.pbn" \
                --ns 1 --ew 1 --n_cards 24 2>&1 | grep -E "opening|later|identical|partscore|game |slam|per board overall") \
                >> "$RUN/eval.txt"
        done
    done
    cat "$RUN/eval.txt"
    log "Done. Models: $MODEL and $INFO"
fi
