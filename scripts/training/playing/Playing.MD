# Training a Neural Network to play

We want the robot to play like humans and not just random based on the double dummy solver, as some of there plays just looks bad (or stupid)

So based on at set of deals we extract the information about the hand, what is seen in dummy, the bidding and ...

Based on this we create a neural network to suggest the card to play.

Input is an array with 298 elements and the output is a card from the deck with 32 different cards.
If the card to lead is a pip the robot selects random.

The input is defined as

