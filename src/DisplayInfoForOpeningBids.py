from nn.models import Models
from nn.bid_info import BidInfo
import binary

#model = BidInfo('../models/21gf_info/binfo_21gf-500000')
model = BidInfo('../models/gib21_info_model/gib21_info-500000')
#model = BidInfo('../models/sayc_info/binfo_sayc-500000')
#model = BidInfo('../models/eamon/21gfinfo/binfo_21gfc-500000')

auction = []

def get_info_for_opening(bid):
    auction = [bid]
    n_steps = 1 + (len(auction)-1) // 4
    ns = -1
    ew = -1
    nesw_i = 1
    vuln = [True, False]
    hand = binary.parse_hand_f(32)("AJT85.AKT.K63.K8")
    A = binary.get_auction_binary(n_steps, auction, nesw_i, hand, vuln, ns, ew)
    p_hcp, p_shp = model.model(A)

    p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
    p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

    def f_trans_hcp(x): return 4 * x + 10
    def f_trans_shp(x): return 1.75 * x + 3.25

    p_hcp = f_trans_hcp(p_hcp)
    p_shp = f_trans_shp(p_shp)

    return p_hcp, p_shp

if __name__ == '__main__':

    p_hcp, p_shp = get_info_for_opening('1C')
    print(f"Opening 1C: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('1D')
    print(f"Opening 1D: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('1H')
    print(f"Opening 1H: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('1S')
    print(f"Opening 1S: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('1N')
    print(f"Opening 1N: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")
 
    p_hcp, p_shp = get_info_for_opening('2C')
    print(f"Opening 2C: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('2D')
    print(f"Opening 2D: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('2H')
    print(f"Opening 2H: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('2S')
    print(f"Opening 2S: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('2N')
    print(f"Opening 2N: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")
    
    p_hcp, p_shp = get_info_for_opening('3C')
    print(f"Opening 3C: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('3D')
    print(f"Opening 3D: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('3H')
    print(f"Opening 3H: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('3S')
    print(f"Opening 3S: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('3N')
    print(f"Opening 3N: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")
 
    p_hcp, p_shp = get_info_for_opening('4C')
    print(f"Opening 4C: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('4D')
    print(f"Opening 4D: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('4H')
    print(f"Opening 4H: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('4S')
    print(f"Opening 4S: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('4N')
    print(f"Opening 4N: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('5C')
    print(f"Opening 5C: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")

    p_hcp, p_shp = get_info_for_opening('5D')
    print(f"Opening 5D: HCP = {p_hcp[0][2]:>4.1f}, Shape = {[round(float(x), 1) for x in p_shp[0][-4:]]}")


