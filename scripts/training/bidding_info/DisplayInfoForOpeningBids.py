import sys

sys.path.append('../../../src')
import argparse
import logging
import os

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import binary
import conf
import bidding
from bidding import bidding

auction = []

def get_info_for_opening(hand, bid, models):
    auction = [bid]
    n_steps = 1
    nesw_i = 1
    vuln = [True, False]
    hand = binary.parse_hand_f(32)(hand)
    A = binary.get_auction_binary(n_steps, auction, nesw_i, hand, vuln, models)
    p_hcp, p_shp = models.binfo_model.model(A)

    p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
    p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

    def f_trans_hcp(x): return 4 * x + 10
    def f_trans_shp(x): return 1.75 * x + 3.25

    p_hcp = f_trans_hcp(p_hcp)
    p_shp = f_trans_shp(p_shp)

    return p_hcp, p_shp

def main():
    parser = argparse.ArgumentParser(description="Process bidding data using a bidder model.")
    parser.add_argument("config_path", help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config_path

    config = conf.load(config_path)
   
    try:
        if (config["models"]['tf_version'] == "2"):
            print("Loading version 2")
            from nn.bidder_tf2 import Bidder
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.bidder import Bidder
            from nn.models import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.bidder import Bidder
            from nn.models import Models


    models = Models.from_conf(config,"..\..\..")

    for bid in range(5,40):
        p_hcp, p_shp = get_info_for_opening("AJT85.AKT.K63.K8",bidding.ID2BID[bid], models)
        print(f"Opening {bidding.ID2BID[bid]}: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    
if __name__ == '__main__':

    main()


