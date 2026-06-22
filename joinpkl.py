import pickle

def load_pkl_records(path):
    records = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = pickle.loads(line)
                records.append(rec)
            except Exception:
                pass
    return records

# Load both files
records1 = load_pkl_records("selfplay_records_13.pkl")
records2 = load_pkl_records("selfplay_records_14.pkl")

# Combine
all_records = records1 + records2

# Write out joined file
with open("selfplay_records_joined.pkl", "wb") as f:
    for rec in all_records:
        f.write(pickle.dumps(rec) + b"\n")

print("Joined", len(records1), "+", len(records2), "records →", len(all_records))
