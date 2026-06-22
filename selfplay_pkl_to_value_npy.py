import pickle
import numpy as np

INPUT_PKL = "selfplay_records_joined.pkl"
OUTPUT_NPY = "selfplay_value.npy"

values = []

with open(INPUT_PKL, "rb") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            rec = pickle.loads(line)
        except Exception:
            continue

        # Skip anything that isn't a dict
        if not isinstance(rec, dict):
            continue

        if "value" in rec:
            v = float(rec["value"])
            values.append(v)

# Keep your original dtype: float32
values = np.array(values, dtype=np.float32)

print("Raw values summary")
print("------------------")
print("Count:", len(values))
if len(values) > 0:
    print("Min:", np.min(values))
    print("Max:", np.max(values))
    print("Mean:", np.mean(values))

np.save(OUTPUT_NPY, values)
print("\nSaved values to:", OUTPUT_NPY)

