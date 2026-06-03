import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
from objects import Card
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer

init()


def _load_dds3():
    """Import the dds3 extension (DDS 3.0.0 Python interface).

    Tries a normal import first (dds3 installed as a wheel). If that fails,
    falls back to a copy vendored under BEN's bin/ directory.
    """
    # Keep the real failure around. A vendored _dds3.so that exists but won't
    # load (e.g. built for a different Python version, or missing a dependent
    # shared library) raises ImportError here; swallowing it hides the cause.
    last_error = None
    try:
        import dds3
        return dds3
    except ImportError as ex:
        last_error = ex

    here = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        plat = "win"
    elif sys.platform == "darwin":
        plat = "darwin"
    else:
        plat = "linux"

    # Look for a vendored dds3 under bin/, covering the layouts BEN runs in:
    #   repo:              src/ddsolver/ddsolver.py -> <repo>/bin       (here/../..)
    #   flattened (Docker): /app/ddsolver/ddsolver.py -> /app/bin       (here/..)
    #   plus BEN_HOME/bin and <cwd>/bin as fallbacks.
    bin_roots = [
        os.path.join(os.path.abspath(os.path.join(here, "..", "..")), "bin"),
        os.path.join(os.path.abspath(os.path.join(here, "..")), "bin"),
    ]
    if os.getenv("BEN_HOME"):
        bin_roots.append(os.path.join(os.getenv("BEN_HOME"), "bin"))
    bin_roots.append(os.path.join(os.getcwd(), "bin"))

    seen = set()
    for root in bin_roots:
        for cand in (os.path.join(root, "dds3-" + plat),
                     os.path.join(root, "dds3"),
                     root):
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import dds3
                return dds3
            except ImportError as ex:
                last_error = ex
                continue

    raise ImportError(
        "Could not import the 'dds3' extension (DDS 3.0.0 Python interface).\n"
        "Build it from the DDS repository and install or vendor it:\n"
        "  bazel build -c opt //python:dds3_wheel_dist   # produces a wheel in dist/\n"
        "  pip install dist/dds3-*.whl\n"
        f"or place the built dds3 package in BEN's bin/dds3-{plat}/ directory.\n"
        f"A _dds3.so built for a different Python than {sys.version_info.major}."
        f"{sys.version_info.minor} will fail to load here.\n"
        "See docs/python_interface.md in the DDS repository.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


dds3 = _load_dds3()


# DDS 3.0.0 removed internal multi-threading from the legacy batch API
# (SolveAllBoards now solves sequentially). The modern model is one
# SolverContext per worker thread; DDSolver parallelises with a thread pool,
# and solve_board_pbn releases the GIL during the solve, so the threads run
# concurrently. Each pool thread keeps its own SolverContext.
_ctx_local = threading.local()


def _thread_context():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        ctx = dds3.SolverContext()
        _ctx_local.ctx = ctx
    return ctx


class DDSolver:

    # Default for dds_mode changed to 1
    # Transport table will be reused if same trump suit and the same or nearly the same cards distribution, deal.first can be different.
    # Always search to find the score. Even when the hand to play has only one card, with possible equivalents, to play.
    # If zero, we not always find the score
    # If 2 transport tables ignore trump

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode
        # max_threads is the size of the solver thread pool.
        # 0 = one thread per CPU core.
        workers = max_threads if (max_threads and max_threads > 0) else (os.cpu_count() or 4)
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dds")
        if verbose:
            sys.stderr.write(f"DDSolver loaded — DDS {self.version()} - dds mode {dds_mode} - {workers} solver threads\n")

    def version(self):
        return "3.0.0"

    def calculatepar(self, hand, vuln, print_result=True):
        with ModelTimer.time_call('dds_par'):
            return self._calculatepar_impl(hand, vuln, print_result)

    def _calculatepar_impl(self, hand, vuln, print_result=True):
        # vulnerable
        # 0: None 1: Both 2: NS 3: EW
        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        # calc_all_tables_pbn computes the DD table and (mode != -1) the par score.
        try:
            result = dds3.calc_all_tables_pbn(["N:" + hand], mode=v)
        except Exception as e:
            sys.stderr.write(f"Error calculating par: {e}, Hand {hand.encode('utf-8')}\n")
            raise

        par_results = result.get("par_results") or []
        if not par_results:
            sys.stderr.write(f"{Fore.RED}No par result for hand {hand.encode('utf-8')}{Style.RESET_ALL}\n")
            return None

        par = par_results[0]
        # par_score is a 2-element list of strings, e.g. "NS 420" / "EW -420".
        ns_par = par["par_score"][0]
        ew_par = par["par_score"][1]

        if print_result:
            print("NS score: {}".format(ns_par))
            print("EW score: {}".format(ew_par))

        return int(ns_par.split()[1])

    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    @staticmethod
    def _trick_number(hands_pbn, current_trick):
        """Derive trick number (1-13) from remaining cards in PBN hand."""
        pbn = hands_pbn[0]
        if ':' in pbn:
            pbn = pbn.split(':', 1)[1]
        remaining = sum(1 for c in pbn if c not in '. ')
        return (52 - remaining - len(current_trick)) // 4 + 1

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        trick = self._trick_number(hands_pbn, current_trick)
        with ModelTimer.time_call(f'dds_solve_t{trick:02d}', items=len(hands_pbn)):
            results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn, solutions)

        return results

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        trump = (strain_i - 1) % 5

        # The current trick is the same for every board in the batch.
        trick_suit = [0, 0, 0]
        trick_rank = [0, 0, 0]
        for i in range(min(3, len(current_trick))):
            trick_suit[i] = current_trick[i] // 13
            trick_rank[i] = 14 - current_trick[i] % 13
        trick_suit = tuple(trick_suit)
        trick_rank = tuple(trick_rank)
        dds_mode = self.dds_mode

        # Solve each board on the thread pool. solve_board_pbn releases the GIL
        # during the DDS search; each pool thread uses its own SolverContext.
        def _solve_one(pbn):
            return dds3.solve_board_pbn(
                pbn,
                trump=trump,
                first=leader_i,
                current_trick_suit=trick_suit,
                current_trick_rank=trick_rank,
                target=-1,
                solutions=solutions,
                mode=dds_mode,
                context=_thread_context(),
            )

        try:
            solved = list(self._pool.map(_solve_one, hands_pbn))
        except Exception as e:
            print(f"{Fore.RED}DDS error: {e} {hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
            return None

        if solutions == 1:
            # Just return the maximum number of the side to play for each sample
            card_results = {}
            card_results["max"] = []
            card_results["min"] = []
            for fut in solved:
                card_results["max"].append(fut["score"][0])
                card_results["min"].append(fut["score"][fut["cards"] - 1])

        else:
            card_results = {}
            for fut in solved:
                for i in range(fut["cards"]):
                    suit_i = fut["suit"][i]
                    card = suit_i * 13 + 14 - fut["rank"][i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(fut["score"][i])
                    eq_cards_encoded = fut["equals"][i]
                    for k, rank_code in enumerate(card_rank):
                        if rank_code & eq_cards_encoded > 0:
                            eq_card = suit_i * 13 + k
                            if eq_card not in card_results:
                                card_results[eq_card] = []
                            card_results[eq_card].append(fut["score"][i])

        return card_results


    def expected_tricks_dds(self, card_results):
        return {card:round((sum(values)/len(values)),2) for card, values in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list : List[float]):
        # Convert to plain Python list to avoid numpy scalar overhead
        probs = [float(p) for p in probabilities_list]
        return {card: round(sum(p*res for p, res in zip(probs, result_list)),2) for card, result_list in card_results.items()}

    def p_made_target(self, tricks_needed):

        def fun(card_results):
            return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        # Create a new dictionary to store sorted counts for each key
        sorted_counts_dict = {}

        # Loop through the dictionary and process each key-value pair
        for key, array in dd_solved.items():
            # Use Counter to count the occurrences of each element
            element_count = Counter(array)

            # Sort the counts by frequency in descending order
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)

            # Store the sorted result in the new dictionary
            sorted_counts_dict[key] = sorted_counts

        # Print the sorted counts for each key
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")
