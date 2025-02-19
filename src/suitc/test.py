import SuitC
import gc
import psutil
suitc = SuitC.SuitCLib(False)

def log_memory_usage():
    # Get system memory info
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
    print(f"Available memory: {available_memory:.2f} MB")
    
log_memory_usage()
for i in range (1000):
    card = suitc.calculate("K3 2 AQJT987654")
    #card = suitc.calculate("AKQJ642 T98753")
    #card = suitc.calculate("A874 KQJ6 T9532")
    if (i+1) % 50 == 0:
        gc.collect()
        log_memory_usage()
