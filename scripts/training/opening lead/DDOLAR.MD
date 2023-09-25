# Training a Neural Network to find opening lead

Training a neural network to find the best opening lead is not simple, but we can use a metric called DDOLAR to evaluate the performance of BEN.

It would appear that the main metrics in use are DDOLAR and ADDOLAR which are defined (by Nicholas Hammond) as follows:

# DDOLAR: Double dummy opening lead accuracy rate (%)

For each lead, look at the double dummy analysis. The lead is either "Good" (does not give up a trick against double dummy) or "Bad" (gives up a trick). For each pair/player, divide the number of good leads by the total number of leads for the DDOLAR.

The term is abbreviated as DDOLAR, pronounced "D-Dollar".

Top experts in IMP play average just under 81%.

ACBL club players in MP play average just under 80%.

ACBL virtual club (VACB) players in MP play average just under 79%.

Any value above 85% is generally suspicious; however you must also look at the number of boards being analyzed. Someone who has a DDOLAR of 90% with 20 boards is far less suspicious than a player with a DDOLAR of 85% over 300 boards.

Examples: In face-to-face play (FTF) Eric Rodwell DDOLAR is 82.07%, Jeff Meckstroth is 80.17%.

Top experts average 81%. ACBL BBO players average 80%. ACBL Virtual Club players average 79%. The range for players with the most data (6,000 total boards played) in ACBL BBO events is 74.60% to 87.45%.

 

# ADDOLAR: Actual double dummy opening lead accuracy rate (%)

ADDOLAR is exactly the same as DDOLAR except that hands where any lead gives the same result are ignored. 

E.g. it makes no difference what card you lead against a stone-cold 7NT

It makes sense to ignore these hands as they simply inflate the DDOLAR across the field (since everyone scores 100%); ADDOLAR reduces the expected values:

Top experts in IMP play average just under 74.7%.

ACBL club players in MP play average about 73.7%.

ACBL virtual club players in MP play average just under 72.7%.

Any value above 80% is suspicious; however you must also look at the number of boards being analyzed. Someone who has a ADDOLAR of 85% with 20 boards is far less suspicious than a player with a ADDOLAR of 80% over 300 boards.

Examples: In face-to-face play (FTF) Eric Rodwell ADDOLAR is 76.42%, Jeff Meckstroth is 73.06%.

# Training BEN

BEN was trained using deals from JACK, and in the current implementation the metrics for BEN is:

- DDOLAR  (suit)  688 = 81.83%
- DDOLAR  (NT)    301 = 76.41%
- ADDOLAR (suit)  518 = 75.87%
- ADDOLAR (NT)    242 = 70.66%

based on 1.000 random deals. Number of deals just before the equal sign.

So the opening lead against suit contracts seems very good, but lead against NT is below average club players.

The deals form JACK had these characteristics

- DDOLAR  (suit) 264963 = 83.61%
- DDOLAR  (NT)   96134 = 73.62%
- ADDOLAR (suit) 191432 = 77.32%
- ADDOLAR (NT)   77182 = 67.15%

So it seems that the training data for NT could be the reason for relativer bad performance.

From JACK we have two different datasets, but with almost the same characteristics. 

- DDOLAR  (suit) 197151 = 83.25%
- DDOLAR  (NT)   75625 = 73.24%
- ADDOLAR (suit) 142945 = 76.89%
- ADDOLAR (NT)   60860 = 66.74%

Trained on this dataset the metrics for BEN is

- DDOLAR  (suit)  691 = 81.77%
- DDOLAR  (NT)    298 = 74.83%
- ADDOLAR (suit)  522 = 75.86%
- ADDOLAR (NT)    239 = 68.62%

Now the questions is if we can improve this, and an idea is to remove all deals from the input, that was an error on the actual board.

By the way if we just pick a random card for opening lead (The monkey test) we will get the following statistics.

- DDOLAR  (suit)  688 = 67.30%
- DDOLAR  (NT)    301 = 64.45%
- ADDOLAR (suit)  518 = 56.56%
- ADDOLAR (NT)    242 = 55.79%


DCWER and DFWER for the 1.000 boards was:


- DCWER  (suit)  691 = 1.06%
- DCWER  (NT)    298 = 2.46%
- DFWER  (suit)  691 = 0.87%
- DFWER  (NT)    298 = 2.03%

Again very fine for suit-play and not good in NT

Using a higher trust for NT-contracts

- DDOLAR  (suit)  698 = 81.81%
- DDOLAR  (NT)    303 = 76.57%
- ADDOLAR (suit)  524 = 75.76%
- ADDOLAR (NT)    244 = 70.90%
