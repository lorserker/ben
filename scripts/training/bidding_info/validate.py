import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python binfo_nn.py inputdirectory ")
    sys.exit(1)

bin_dir = sys.argv[1]
X_train = np.load(os.path.join(bin_dir, 'X.npy'))
HCP_train = np.load(os.path.join(bin_dir, 'HCP.npy'))
SHAPE_train = np.load(os.path.join(bin_dir, 'SHAPE.npy'))

#x = np.zeros((rows_pr_hand * n, 8, 161), dtype=np.float16)

def hand_to_str(hand):
    x = hand.reshape((4, 8))
    symbols = 'AKQJT98x'
    suits = []
    for i in range(4):
        s = ''
        for j in range(8):
            if x[i,j] > 0:
                s += symbols[j] * int(x[i,j])
        suits.append(s)
    return '.'.join(suits)

print(hand_to_str(X_train[0,1,9:41]))  
for i in range(8):
    print(np.argmax(X_train[0,i,41:81]),np.argmax(X_train[0,i,81:121]),np.argmax(X_train[0,i,121:161]))  
for i in range(8):
    print(HCP_train[0,i] * 4 + 10)
    print(SHAPE_train[0,i] * 1.75 + 3.25)