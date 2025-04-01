import SuitC
import gc
import psutil
suitc = SuitC.SuitCLib(False)

# Get system memory info
virtual_memory = psutil.virtual_memory()
available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
print(f"Available memory: {available_memory:.2f} MB")

#-F1 -u -c100 -ls6 KT63 972 AQJ854
card = suitc.calculate(4, "KT63", "972", "AQJ854")

print(card)
