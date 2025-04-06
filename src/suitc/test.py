import SuitC
import gc
import psutil
import time
suitc = SuitC.SuitCLib(True)

# Get system memory info
virtual_memory = psutil.virtual_memory()
available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
print(f"Available memory: {available_memory:.2f} MB")

t_start = time.time()

#-F1 -u -c100 -ls6 KT63 972 AQJ854
#card = suitc.calculate(4, "KT63", "972", "AQJ854")
card = suitc.calculate(4, "K83", "AJ942", "QT765")

print(card)

print(f"Estimating took {(time.time() - t_start):0.4f} seconds")