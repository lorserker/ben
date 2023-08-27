import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy.special import softmax


SUIT_MASK = np.array([
    [1] * 8 + [0] * 24,
    [0] * 8 + [1] * 8 + [0] * 16,
    [0] * 16 + [1] * 8 + [0] * 8,
    [0] * 24 + [1] * 8,
])


class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.graph.finalize()
        self.model = self.init_model()

    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')  #  we always give the whole sequence from the beginning. shape = (batch_size, n_tricks, n_features)
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')  #  shows which card it would play at each trick. (but we only care about the card for last trick)

        p_keep = 1.0

        def pred_fun(x):
            result = None
            with self.graph.as_default():
                card_logit = self.sess.run(out_card_logit, feed_dict={seq_in: x, keep_prob: p_keep})
                result = self.reshape_card_logit(card_logit, x)
            return result

        return pred_fun

    def reshape_card_logit(self, card_logit, x):
        return softmax(card_logit.reshape((x.shape[0], x.shape[1], 32)), axis=2)

    def next_cards_softmax(self, x):
        return self.model(x)[:,-1,:]


class BatchPlayerLefty(BatchPlayer):

    def reshape_card_logit(self, card_logit, x):
        return softmax(card_logit.reshape((x.shape[0], x.shape[1] - 1, 32)), axis=2)


def follow_suit(cards_softmax, own_cards, trick_suit):
    assert cards_softmax.shape[1] == 32
    assert own_cards.shape[1] == 32
    assert trick_suit.shape[1] == 4
    assert trick_suit.shape[0] == cards_softmax.shape[0]
    assert cards_softmax.shape[0] == own_cards.shape[0]

    suit_defined = np.max(trick_suit, axis=1) > 0
    trick_suit_i = np.argmax(trick_suit, axis=1)

    mask = (own_cards > 0).astype(np.int32)

    has_cards_of_suit = np.sum(mask * SUIT_MASK[trick_suit_i], axis=1) > 1e-9

    mask[suit_defined & has_cards_of_suit] *= SUIT_MASK[trick_suit_i[suit_defined & has_cards_of_suit]]

    legal_cards_softmax = cards_softmax * mask

    s = np.sum(legal_cards_softmax, axis=1, keepdims=True)
    s[s < 1e-9] = 1

    return legal_cards_softmax / s

def get_trick_winner_i(trick, strain):
    assert trick.shape[1] == 4 * 32
    assert strain.shape[1] == 5

    n_samples = trick.shape[0]

    trick_cards = np.hstack([
        np.argmax(trick[:,:32], axis=1).reshape((-1, 1)),
        np.argmax(trick[:,32:64], axis=1).reshape((-1, 1)),
        np.argmax(trick[:,64:96], axis=1).reshape((-1, 1)),
        np.argmax(trick[:,96:], axis=1).reshape((-1, 1))
    ])
    assert trick_cards.shape == (n_samples, 4)

    trick_cards_suit = trick_cards // 8

    trump_suit_i = np.argmax(strain, axis=1).reshape((-1, 1)) - 1

    is_trumped = np.any(trick_cards_suit == trump_suit_i, axis=1).reshape((-1, 1))

    trick_cards_trumps = trick_cards.astype(float)
    trick_cards_trumps[trick_cards_suit != trump_suit_i] = 99
    trick_cards_trumps += np.random.randn(n_samples, 4) / 1000  # adding random to break ties
    highest_trump_i = np.argmin(trick_cards_trumps, axis=1).reshape((-1, 1))

    lead_suit = trick_cards_suit[:,0].reshape((-1, 1))

    trick_cards_lead_suit = trick_cards.astype(float)
    trick_cards_lead_suit[trick_cards_suit != lead_suit] = 99
    trick_cards_lead_suit += np.random.randn(n_samples, 4) / 1000  # adding random to break ties
    highest_lead_suit_i = np.argmin(trick_cards_lead_suit, axis=1).reshape((-1, 1))

    return np.where(is_trumped, highest_trump_i, highest_lead_suit_i)
