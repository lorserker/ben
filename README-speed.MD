# Do you want speed or quality?

When configuring BEN, there are many considerations to make, do we want speed or quality? The problem might be that when BEN is to slow users get tired of waiting, and not all the time spent will improve the actual bidding or play.

BEN is using Neural networks, that are very fast, and given the proper training also quite precise, but to help we have added different functionality, and here is a list of those features, and help for you to decide how you want to configure your robot.

# Bidding

When bidding the first step is to ask the neural network what to bid given the hand and context, and the neural network will respond with all bids and a score between 0 and 1 for each bid of how well the bid match the hand - the sum of all bids will be 1.0

If you want a speedy bidding bot you can just accept the best bid and BEN will bid most hands between 1 and 2 seconds - about 0.1 sec pr. bid.

Now if you want to trade some speed for quality you can configure

# If there are multiple bids over this threshold make a simulation for the bids
search_threshold = 0.07

Now if BEN find 2 or more bids above this value it will start a search for the right bid, by bidding the sample hands to the end, and then evaluate the score for the final contracts, and again we hit a point where we can trade quality for speed.

In the original BEN a neural network, trained on data from Jack, was used to estimate the score based on the hands. This was very fast but not so good at highlevel contracts (or bad contracts), so a new Neural Network was developed using 10 mill deals from Pavlicek, and finally we added real double dummy simulation to find the score.
Double dummy seems to give the best result but is more time consuming.
A major difference between the two neural networks are that the original needs the opening lead to give an estimate, so this parameter is used to define which net is used

# use the following estimator, sde, dde or both
estimator = dde

If you want the slower Double dummy then this need to be activated

# Use double dummy when estimating tricks during bidding - to slow if 200 samples
double_dummy_calculator = True

Now the speed for bidding is about 10-15 sec pr board or about 1-2 sec pr. bid on average
The calculation of speed is based on this parameter:

# Number hands where we will calculate scores for when bidding
sample_hands_auction = 200

So changing that could improve speed (or decrease), but 200 seems to be a fine number.

Ok, now the bidding is over so let us look at the opening lead

Again BEN starts by asking a neural network, what to lead based on samples, so first we need to look at the number of samples generated.

# How many boards should we sample to find the number of samples below
sample_boards_for_auction_opening_lead = 5000

This is the number we generate and the try to match the bidding. On average about 1% will match so leaving us with 50 samples to use for finding the best lead.

If we increase the number of samples we might hit another limit

# Number of samples made, when finding the opening lead
sample_hands_opening_lead = 200

as we are setting a max for the number of samples we will caculate the best lead on


We also ask the neural network given our hand and information extracted from the bidding, what it think we should lead, and we will ignore leads below this threshold:

# Ignore cards as opening lead below this value from the neural network
lead_threshold = 0.50

but we make sure we find at least one opening lead
# Force a minimum number of leads to consider - overrides lead_threshold
min_opening_leads = 1

And finally we have to decide how we want to validate the opening lead

# Use double dummy statistics when evaluating the opening lead - default is single dummy
double_dummy = False

Where we can use a neural network or real double dummy - the neural network is the one mentioned above besed on data from Jack

So going for speed BEN will normally find the opening lead within a second, and going for quality it can take 10-20 seconds

After the opening lead we will have a look at the play.

Again BEN is using a neural network to assist, but the network is not good enough to standa alone, so each play is simulated using samples, and there is caclulated a double dummy result.

So each play is selected based on double dummy and neural network, you can see it as double dummy finds a set of cardfs, and the neural network then selects the card that seems right based on the training.

It is difficult to impprove this performance wise, but Lorand has implemented a solution on BBO based on only the neural network, again speed over quality.

To improve performance these settings can be reduced

# Number of samples when playing the hand
sample_hands_play = 200
# Number of samples we will generate to find sample_hands_play
sample_boards_for_play = 5000

The last is reduced during the play when less cards remain.

But in total BEN will now play a board in about 30 sec, and most cards are played in less than 0.1 sec

But to improve the play we have also added a new play engine Perfect Information Monte Carlo (PIMC), that is based on generating all deals that match criterias for suit lengths and honor points. This is much slower, but better. We have seen that during the first tricks BEN is better creating good samples, and PIMC is not capable of testing all possible combinations, so we have added another configureation:

# Are we using PIMC as declarer or defender? 
pimc_use_declaring = True
pimc_use_defending = True
# Max wait time for results from PIMC in seconds
pimc_wait = 3
# When should PIMC kick in
pimc_start_trick_declarer = 4
pimc_start_trick_defender = 8

Now from trick 4 declarer will use PIMC and with the above setting typical use up to 3 seconds to select a play, but from trick 8, where the defence also use PIMC, the repsponse times are almost the same as PIMC can mange to handle all combinations, and as BEN won't generate samples it will be quicker most of the time. In total it is about 10-20 seconds more pr. board.

The current default.conf contains settings, where I have tried to create the best setup taking in consideration the robot should still be fun to play with, and in that setup BEN uses about 1 min and 30 seconds pr. board.

In the configuration file, there are also many other interesting settings, but I think I have covered most of the parameters that is related to the speed/quality problem.

Have fun building your own configuration.




