# Training a Neural Network to Bid

Using a neural network to bid is implemented using a neural network, where input is the given context, and output is the actual bid.

Let us start with the simplest first as output is a bid from 1C to 7N, Pass, Double or Redouble or 38 possible bids.

The context is more complex and consist of
1: The hand
2: Vulnerability
3: Bidding until now
4: Opponents system

The hand is always 13 out of 52 cards, and vulnerability has one of 4 different values

Bidding is a sequence of bids, and almost unlimited (2^35). Fortunately most of the bidding sequences can be ignored, so we can focus on the more common sequences. Looking thru the championships, there was a bidding sequence of 43 bids in the European championship in 2004, but it was a strong Club system, and that is not expected that this implementation will be able to handle that kind of system (The system is very specific, with many relays, not well suited for a neural network).

Lorand made a decision, that BEN will only be able to handle bidding sequences up to a length of 32, and decided, that the bidding should be represented by a bidding round (one bid from each player), so in total 8 bidding rounds.

To signal, that the bidding was ended a new bid was added "PADDING_END", and there is a need for having a "bid" before the auction starts, "PADDING_START" (explained more later), so we ended up with 40 bids as possible output, where the 2 new bids never should occur.

The vulnerability was implemented as 2 booleans, one for each side

The hand was simple to represent, but to help the neural network, the shape and hcp for the hand was calculated and added to the input. As small cards doesn't have any value in the bidding the set of cards was reduced to 32.

Until now the opponents system is not in use, but a system could be represented by a number, and it is possible to create a specific neurtal network, that should be used against a specific systrem (including conventions).

But let us see at the actual implementation looking at a deal like this:

- E None AJ64.9865.9.Q987 Q7.AT43.QT3.AT63 982.J2.A542.KJ42 KT53.KQ7.KJ876.5

Dealer East, could probably be ignored.

Vulnerability None is represented as to booleans [False, False] telling both sides are non-vulnerable.

Hand for each of the players is represented as an array with 32 elements, and Norths hand in this example is

- KT53.KQ7.KJ876.5

and is translated to:

- 0 1 0 0 1 0 0 2 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 2 0 0 0 0 0 0 0 1

So starting from Ace of spades, and each suit ends with a value for number of small cards from 2-7.

In a deck there is 40 high card points (hcp), and the hcp is calculated for each hand. The values are linear normalized.

The shape of a hand is the length in the 4 suits, and again it is normalized (also known as z-score or standardization)

So we end up representing the hand as

- [ 0, 0, 0.5, 0.43, -0.11, 1.03, -1.29, 0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1]

So we are calculating hcp and shape so the neural network will not have to build neurons for this, and we are condensing information about the hand like small cards, to remove the information as it is not being used in the bidding.

There are a lot of other metrics for a hand, that is used in bidding, and it might be a good idea to help the neural network with those also, but if the input is not providing deals where this information is relevant it will just create an overhead.

It is also important to realize, that a lot of that type of information is used at the later bidding stages, and should not be used in the first bidding rounds.

The following metrics should be condsidered

- Losers
- Controls
- Keycards
- Suit quality
- Stoppers

But this will also reduce what the neural network is learning, as players are not using the same rules for defining suit quality, stoppers etc, and we would like the network to learn that stuff like suit quality is a factor.

For the training i don't expect we will have enough data, where it will make a difference for the outcome if a suit is changed from

- KJ986 to KJ432

especially not if we add suit qualiy as a metric, so reducing the card set to 24 or even 20 is probably a possible solution.

So now left is the bidding, and Lorand decided that each bidding round should be represented as input data, and it was then obvious to use a Recurrent neural network.

Seen from a specific hand a bidding round consist of bid from LHO (Left Hand Opponent), Partner and RHO (Right Hand Opponent)

When the player is the first to bid, we introduce the new bid "PADDING_START", for the 3 other player. I reality just telling, that there was no bid in the other positions.

A single bid is represented as an array with 40 elements, where only 1 item is 1, and the rest is zero (a One-Hot array), as this will help the neural network. The values of the Hot-array is

0 = PAD_START
1 = PAD_END
2 = PASS
3 = Double
4 = Redouble
5-39 = Bids, starting from 1C and ending at 7N,

So we now have the input record for a bid as this (Q7.AT43.QT3.AT63)

[ 0, 0, 0.5, -0.71, 0.43, -0.11, 0.43,  
   0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 2,  
   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

3 times PAD_START just means that the player is first hand to bid.

And the expected output as this:

[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

as the hand would open 1C (Element 5 is hot).

Now as we have a deal with 4 hands, there are 4 hands we can train at the same time, we end up with a 3-dimensional array.

In the current implementation (using a Recurrent Neural Network) the bids are as mentioned divided in bidding rounds, so there is an implementation of state, that holds the information about the actual bidding round, so the neural network get that information, when predicting the next bid. So to describe the bidding input for one hand we have 8 times 3 bids or 240 elements in 3 one-hot arrays.

To avoid using state it could be possible to define that bidding consist of up to 50 bids in a sequence. The number of records for the bidding would be limited to the actual number of bids on a hand - probably about 4, but still we would use 4*50*40 elements in one-hot arrays.

But I am not sure about we then will be removing the neural networks ability to generalize, but it is worth testing.














