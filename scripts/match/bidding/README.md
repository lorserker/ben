# Bidding Matches

This describes how to run ben vs ben matches where each deal stops after the opening lead and the double summy score after the lead is used as the result.

Different models are provided using different configuration files, for example [A.conf](A.conf) and [B.conf](B.conf)

It's not recommended to run very different systems against each other, because the engine still assumes that NS and EW are playing the same or very similar systems.

Matches can be run with or without search during the bidding. Without search is much faster because it just uses the output of the neural network and doesn't use any simulation at all.

Following are the steps to run a match:

Set the environment variable `BEN_HOME` to ben root directory

```
export BEN_HOME=../../..
```

Generate a new set of deals (giving the number of deals as an argument)

```
python create_set.py 32 > set0001.txt 
```

Run the auctions for every deal at the first table.

```
python auction.py \
    --bidderNS=A.conf \
    --bidderEW=B.conf \
    --set=set0001.txt \
    > auctions1.json
```

Run the auctions for every deal at the second table (flipping the values of the `--bidderNS` and `--bidderEW` arguments)

```
python auction.py \
    --bidderNS=B.conf \
    --bidderEW=A.conf \
    --set=set0001.txt \
    > auctions2.json
```

If you want to enable search during the bidding, just add the `--search` argument like this:

```
python auction.py \
    --bidderNS=B.conf \
    --bidderEW=A.conf \
    --set=set0001.txt \
    --search \
    > auctions2.json
```

After the bidding, add the opening leads:

```
cat auctions1.json | python lead.py > leads1.json
cat auctions2.json | python lead.py > leads2.json
```

Score the outcome (double dummy)

```
cat leads1.json | python score.py > results1.json
cat leads2.json | python score.py > results2.json
```

Finally compare the results from the two tables to see who has done better.

```
python ..\..\..\src\compare.py results1.json results2.json > compare.json
```

The output file will contain the comparison and result (IMPs) for each deal from the set (one deal per line) as json objects. An example of such a comparison object is:

```
{
  "dealer": "N",
  "vuln": "E-W",
  "north": "T543.3.KQ8543.AK",
  "east": "AKQ96.A862.2.J96",
  "south": "8.KJT5.JT97.Q842",
  "west": "J72.Q974.A6.T753",
  "auction": [
    [
      "1D",
      "1S",
      "X",
      "2S",
      "PASS",
      "PASS",
      "PASS"
    ],
    [
      "1D",
      "1S",
      "X",
      "2S",
      "PASS",
      "PASS",
      "X",
      "PASS",
      "3D",
      "PASS",
      "PASS",
      "PASS"
    ]
  ],
  "contract": [
    "2SE",
    "3DN"
  ],
  "lead": [
    "DJ",
    "SA"
  ],
  "dd_tricks": [
    8,
    10
  ],
  "dd_score": [
    -110,
    130
  ],
  "imp": -6
}
```

## Optional but useful commands

To run all the steps for a match in a script

```
source match.sh set0001.txt compare.json
```

Matches of 10000 boards take about an hour to complete (if search is not enabled). Matches of 100000 boards can be run overnight.

To see the progress of the match (i.e how many deals have been bid already, etc.)

```
watch 'ls *.json | xargs wc -l' 
```

To filter for boards based on the number of imps (for example to look at boards which aren't push boards):

```
cat compare.json | jq -c 'select(.imp != 0)'
```

To add up the total IMPs

```
cat compare.json | jq .imp | awk '{s+=$1} END {print s}'
```
