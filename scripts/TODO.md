- Add signals when defending
- When declaring we want the declaring CardPlayer to play for Dummy (Sort of sharing knowledge)
- Play on ben.aalborgdata from a uploaded PBN-file
- When bidding rollout, consider expanding the tree for more that the best bid if other bids above threshold



- Lefty in board 31 Camrose deals
- Consider ignoring doubles
- If dummy and pip always play lowest
- Add Hint for playing (Show the card BEN would have played)
- If only low scores from nn, consider other (final) contracts
- Allow the GUI to disable simulation during bidding
- Chakc that when void is know all samples are according to that knowledge

- Consider using SVG-cards for GUI
- Switch between the different simulation engines depending on level of bidding
- Consider disabling simulation twhen responding to 4N

- When rollout bidding, chack for bridge illogical sequences: 1H-2C-X-P-P-3C

- Adding probability of bidding to the double dummy estimates

- Rollout bidding should probably estimate based on how likely it matches partners hand (Camrose board 9)


 Had a few distractions here but should have answered sooner anyway sorry. I've been on the fence about bridgebots, whether to try and be a part of the current project or do something from the ground up. It's always tempting to go for the latter but there's a lot of work.

Bidding and play are pretty separate, but if you're going to do either first, it should be play, because bidding is likely to use play (for simulation) a lot more than play uses bidding.

As far as I know, the current bots all use the same basic idea:

— Deal out a bunch of hands consistent with what we know so far
— Solve them all double-dummy
— Play the card which does the best over all the hands

This is all right as far as it goes but it has some big blind spots.

The most important problem is that you're assuming you're going to play perfectly for all subsequent cards.

Consider this hand for example:

QJxxx
Qxx
AQ
KJ10

AKxxx
AKx
xx
Axx

6S, imps, heart lead.

Any non-beginner human is going to draw trumps, cash the hearts, play Ace & Queen of diamonds and claim.

The computer is going to take the heart finesse, because it thinks "I'm going to pick up the clubs no problem later, so I might as well take the heart finesse for a possible overtrick."

This is really a "test case", in my opinion. If your play engine gets this wrong, it's old school. If it gets it right, it's getting somewhere. (Maybe engines now do get this right, and I'm just behind the times?)

Another big problem with GIB (at least, as it used to be on BBO) is that it doesn't assume opponent is sane.

For example:

Ax
-
AKQJ1098
Axxx

Kxxx
xxx
xxx
Kxxx

3NT, imps, HK lead. (I used to make lots of worse contracts than this.)

The play is to throw dummy's Ace of spades away. Now West re-evaluates and often switches to a spade. It thinks "well, maybe declarer has a heart stop in his hand and not a spade stop, and the spades are now wide open". It is (was) incapable of thinking "declarer would never do that if he had a heart stop and no spade stop in his hand."


It seems that any half-way decent play bot has to use simulated, authentic play, not double-dummy play, to see what might happen, or to decide what hands fit what HAS happened. The trouble is this gets very recursive very fast. Every new card, you say OK, let's sim from here, so we deal 20 hands, and another 20 for every one of those, and so on . . . just impossible.

So you need some sort of clever algorithm that somehow combines Monte Carlo stuff with "realistic" simulation.

A playbot of this nature might, for example, play for a misdefence rather than take a small "legitimate" chance, if the misdefence is more likely. (And this sort of assumption is pretty much essential for decent defence. "If declarer had A, he would have done X, but he actually did Y, so he must have B." You might miss out on the chance of punishing idiots on occasion but you'll never get swindled. You'll basically always achieve par.)


Consider adding a penalty for switching suit - this is a good rule for any bridge player.

Use Neural Network to check if we missed a game / slam

New version of the play neural network has an error where it is playing Q with QT9 over the jack, so we are using the old net