import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

n_cards = 32
pips = 7
symbols = 'AKQJT98x'
n_cards = 24
pips = 9
symbols = 'AKQJTx'

LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUITS = ['C', 'D', 'H', 'S', 'N']
SUIT_RANK = {suit: i for i, suit in enumerate(SUITS)}

BID2ID = {
    '--': 0,
    'PAD_END': 1,
    'PASS': 2,
    'X': 3,
    'XX': 4,
}

SUITBID2ID = {bid: (i+5) for (i, bid) in enumerate(
    ['{}{}'.format(level, suit) for level in LEVELS for suit in SUITS])}

BID2ID.update(SUITBID2ID)

ID2BID = {bid: i for i, bid in BID2ID.items()}


def hand_to_str(hand):
    #print(hand)
    x = hand.reshape((4, n_cards // 4))
    suits = []
    for i in range(4):
        s = ''
        for j in range( n_cards // 4):
            if x[i,j] > 0:
                s += symbols[j] * int(x[i,j])
        suits.append(s)
    return '.'.join(suits)

def print_input(x, y, z):
    #print(x.shape, y.shape, z.shape)
    #print(x[0,0:9])
    for i in range(x.shape[0]):
        if i > 0:
            continue
        bid = np.argmax(y[i])
        if bid > 1:
            if i == 0: 
                print(hand_to_str(x[i,9:9+n_cards]), end="")
            for k in range(4):
                print(ID2BID[np.argmax(x[i,9 + n_cards + k*40:81+k*40])],end=" ")

            print("bid ",end="")                
            print(ID2BID[np.argmax(y[i])],end="")
            if (z[i,0] != 0):             
                print("*",end="")
            print()

# Load the saved model
X_train = np.load('./24Cards/X.npy')
y_train = np.load('./24Cards/y.npy')
z_train = np.load('./24Cards/z.npy')

#print(X_train.shape)
#print(y_train.shape)
#print(z_train.shape)

for i in range(0, X_train.shape[0],4):
    print_input(X_train[i, :, :], y_train[i, :, :], z_train[i, :, :] )

