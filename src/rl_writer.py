import pickle

class RLWriter:
    def __init__(self, filename):
        self.f = open(filename, "ab")

    def write(self, features, policy_target, value_target):
        record = {
            "features": features,
            "policy": policy_target,
            "value": value_target,
        }
        self.f.write(pickle.dumps(record) + b"\n")
        self.f.flush()   # <-- THIS FIXES EVERYTHING

    def close(self):
        self.f.close()
