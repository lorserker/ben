# Bridge Engine

This is a game engine for [bridge](https://en.wikipedia.org/wiki/Contract_bridge).
It can be used as a robot player, to do analysis, or to develop AI for bridge.

The engine is built using machine learning (neural networks) and [double dummy solvers](https://github.com/dds-bridge/dds) through the python wrapper [python-dds](https://github.com/Afwas/python-dds).

To get a first impression of how the engine plays, please take a look at the [demo](https://lorserker.github.io/ben/demo/demo.html).

For a few examples of how you can use the engine for analysis or development, see the [tutorials](#tutorials).

If you like the software, please consider making a donation to support the developers.

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?business=Z7G6CCTFK2XXS&no_recurring=0&currency_code=EUR)

## Getting Started

### Installation

The engine runs on Python 3.7, with the neural networks using tensorflow 1.15

The recommended way of installing is in a [conda environment](https://docs.conda.io/en/latest/miniconda.html). After installing conda, execute the commands in the [conda_setup.sh](conda_setup.sh) script to create an environment and install the dependencies.

The bridge engine was developed on linux. It will likely work on any other platform which runs Python and conda, but was only tested on Windows 10 and Windows 11.

### Running the Web App

The engine has a (very basic) UI which enables you to play as a human agains three bots and then review the games you played.

The app runs in the browser, and the service has two components: (1) "appserver" which serves the UI through http, and (2) "gameserver" which serves the API to interface with the bots through websockets.

Following are instructions to start the service:

First, make sure that you are located in the `src` directory

```bash
cd src
```

Activate the conda environment

```bash
conda activate ben
```

Start the game server

```bash
sudo python gameserver.py
```
And start the app server

```bash
sudo python appserver.py
```

Now you can use the browser to navigate to `http://127.0.0.1:8080/home`
There will be a link "Play now". If you click that, a new board will be started and you can start playing bridge (you are South).
After playing a hand, a link to an analysis page will appear on the home page. The UI was tested on Firefox, Chrome and Android.

![image](ben_screenshot.png)


### Tutorials

The tutorials serve as illustration of the engine's API and functionality, and should be helpful as a starting point if you would like to use the engine in a more customized way for analysis or for development.

The examples run in jupyter notebooks, but it's also possible to just read them without installing anything.

- __[Bidding](src/examples/Bidding.ipynb)__ shows how you can use the engine to bid hands and to sample hands which are consistent with a given auction
- __[Opening Leads](src/examples/OpeningLead.ipynb)__ illustrates how the engine "decides" which card to lead
- __[Card-by-card Analysis](src/examples/CardByCardAnalysis.ipynb)__ this example shows how the engine can go through an already layed hand, analyse every bid made and every card played, and find possible mistakes. You can also give it a hand you played on BBO to see what the engine "thinks" about your play
- __[Single Dummy Estimates](src/examples/SingleDummyEstimates.ipynb)__ this exemplifies the use of a neural network to predict how many tricks declarer will take in a given contract on a given opening lead. As opposed to double dummy solvers, the results are approximate, but much faster.

## Play against itself and against other bots

To make the engine play against itself, execute this command:

```bash
sudo python game.py
```

After each board, it's possible to go through the play using the web UI.

To make the engine play against other bots:

First run the [bridge monitor](http://www.wbridge5.com/bm.htm) table manager server.

Then, run the command below to connect to the table manager using the [blue chip bridge protocol](https://web.archive.org/web/20210514012054/http://www.bluechipbridge.co.uk/protocol.htm)

```
python table_manager_client.py <table-manager-IP> <port> ben West
```

that will connect to the table as West; the other seats can be taken similarly.

## What the engine cannot do (yet)

Below is a list of features which aren't implemented in this engine yet, but you would probably expect them from a bridge robot:

- can't play any other system except 2/1 (see [convention card](convention_card.md))
- can't configure the bidding system and the conventions (it just plays what it plays and assumes the opponents play that as well)
- can't alert
- can't explain bids or understand explanations of bids
- can't use or interpret defensive signals
- can't claim or accept claims
- it does reach good slams, but slam auctions are a bit weird
- isn't deployed as a service which you could play with on the internet (you have to install it to use it)

## How to cite

```
@misc{Dali2022,
  author = {Dali, Lorand},
  title = {Bridge Engine},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lorserker/ben}},
}
```
