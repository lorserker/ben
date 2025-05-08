import tensorflow as tf
import numpy as np
import os
import colorama

colorama.init(autoreset=True)

np.set_printoptions(precision=3, suppress=True, linewidth=300,threshold=np.inf, formatter={'float': '{:7.3f}'.format})

SUIT_MASK = np.array([
    [1] * 8 + [0] * 24,
    [0] * 8 + [1] * 8 + [0] * 16,
    [0] * 16 + [1] * 8 + [0] * 8,
    [0] * 24 + [1] * 8,
])

def remove_cards_not_in_hand(cards_softmax, own_cards, trick_suit, n_cards=32):
    assert cards_softmax.shape[1] == n_cards
    assert own_cards.shape[1] == n_cards
    assert trick_suit.shape[1] == 4
    assert trick_suit.shape[0] == cards_softmax.shape[0]
    assert cards_softmax.shape[0] == own_cards.shape[0]

    suit_defined = np.max(trick_suit, axis=1) > 0
    trick_suit_i = np.argmax(trick_suit, axis=1)
    mask = (own_cards > 0).astype(np.int32)
    has_cards_of_suit = np.sum(mask * SUIT_MASK[trick_suit_i], axis=1) > 1e-9
    mask[suit_defined & has_cards_of_suit] *= SUIT_MASK[trick_suit_i[suit_defined & has_cards_of_suit]]
    legal_cards_softmax = cards_softmax * mask * 100
    s = np.sum(legal_cards_softmax, axis=1, keepdims=True)
    s[s < 1e-9] = 1
    return legal_cards_softmax / s

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

def decode_card(card32):
    suit_i = card32 // 8
    card_i = card32 % 8
    return 'SHDC'[suit_i] + 'AKQJT98x'[card_i]

# Load the trained model
model_path =  'model/NT_2024-12-31-E010.keras'  # Replace with your model path
model = tf.keras.models.load_model(model_path)

bin_dir = 'lead_keras_suit'
X_test = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
B_test = np.load(os.path.join(bin_dir, 'B.npy'), mmap_mode='r')
y_test = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')

n_samples = 1000 # X_test.shape[0]
# Get the first element from X and B
X_sample = X_test[0:n_samples,:]
B_sample = B_test[0:n_samples,:]

# Get the actual value from y
y_actual = y_test[0:n_samples,:]


#for i in range(n_samples):
#    print(X_sample[i])
#    print(hand_to_str(X_sample[i,10:42]))
#    print(B_sample[i])

# Predict using the model
y_pred = model.predict({'X_input': X_sample, 'B_input': B_sample},verbose=1)

# Compare the predicted value with the actual value
predicted_label = np.argmax(y_pred, axis=1)
actual_label = np.argmax(y_actual, axis=1)

#print(f"Predicted label: {predicted_label}")
#print(f"Actual label: {actual_label}")

right_predictions = 0
wrong_predictions = 0
target_shape = y_pred.shape[0]

# Repeat the last parameter so it matches the first dimension of y_pred
last_param = np.array([[0, 0, 0, 0]])
expanded_last_param = np.repeat(last_param, target_shape, axis=0)

# Now, call the function with the updated parameter
y_pred = remove_cards_not_in_hand(y_pred, X_sample[:, 10:42], expanded_last_param)

for i in range(n_samples):
    if predicted_label[i] != actual_label[i]:
        if y_pred[i][actual_label[i]] < 0.05:
            print( f"{colorama.Fore.RED}{hand_to_str(X_sample[i,10:42])} Wrong prediction {decode_card(predicted_label[i])} {y_pred[i][predicted_label[i]]:.3f} {decode_card(actual_label[i])} {y_pred[i][actual_label[i]]:.3f} {colorama.Style.RESET_ALL}")
            wrong_predictions += 1
        else:
            right_predictions += 1

print(f"Correct predictions: {right_predictions} of {n_samples} - wrong {wrong_predictions}")