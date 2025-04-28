import SuitC
import gc
import psutil
import time
suitc = SuitC.SuitCLib(False)

# Get system memory info
virtual_memory = psutil.virtual_memory()
available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
print(f"Available memory: {available_memory:.2f} MB")

t_start = time.time()

hand = "N:K2.K643.854.J762 T974.Q975.QJ2.83 AQ8.AT.AKT73.AKQ J653.J82.96.T954"
hands = hand[2:].split(' ')
declarer = hands[2]
dummy = hands[0]
tricks = suitc.get_trick_potential(declarer, dummy)

print(tricks)

print(f"Estimating took {(time.time() - t_start):0.4f} seconds")