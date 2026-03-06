import sys
import ctypes
from typing import List
from collections import Counter
from objects import Card
from ddsolver import ddss
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer

init()

class DDSSolver:
    """Double Dummy Super Solver — drop-in replacement for DDSolver.

    Uses the DDSS library (7x faster fork of DDS 2.9.0) when available.
    Falls back transparently to the original DDSolver if ddss.dll/libddss.so
    is not found.

    Usage:
        from ddsolver.ddssolver import DDSSolver
        solver = DDSSolver(dds_mode=1, max_threads=0, verbose=True)
        # same API as DDSolver
    """

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode

        if ddss.DDSS_AVAILABLE:
            ddss.SetMaxThreads(max_threads)
            self._fallback = None
            if verbose:
                ver = self._get_ddss_version()
                sys.stderr.write(f"DDSSolver loaded — DDSS {ver} — dds mode {dds_mode} — max threads {max_threads}\n")
        else:
            # Fall back to original DDSolver
            from ddsolver.ddsolver import DDSolver
            self._fallback = DDSolver(dds_mode, max_threads, verbose)
            if verbose:
                sys.stderr.write("DDSSolver: DDSS not available, using DDSolver fallback\n")

    # ------------------------------------------------------------------
    # Public API — identical to DDSolver
    # ------------------------------------------------------------------

    def version(self):
        if self._fallback:
            return self._fallback.version()
        return self._get_ddss_version()

    def calculatepar(self, hand, vuln, print_result=True):
        if self._fallback:
            return self._fallback.calculatepar(hand, vuln, print_result)
        with ModelTimer.time_call('ddss_par'):
            return self._calculatepar_impl(hand, vuln, print_result)

    @staticmethod
    def _trick_number(hands_pbn, current_trick):
        """Derive trick number (1-13) from remaining cards in PBN hand."""
        pbn = hands_pbn[0]
        if ':' in pbn:
            pbn = pbn.split(':', 1)[1]
        remaining = sum(1 for c in pbn if c not in '. ')
        return (52 - remaining - len(current_trick)) // 4 + 1

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        if self._fallback:
            return self._fallback.solve(strain_i, leader_i, current_trick, hands_pbn, solutions)
        trick = self._trick_number(hands_pbn, current_trick)
        with ModelTimer.time_call(f'ddss_solve_t{trick:02d}', items=len(hands_pbn)):
            results = self._solve_helper(strain_i, leader_i, current_trick,
                                         hands_pbn[:ddss.MAXNOOFBOARDS], solutions)
            if len(hands_pbn) > ddss.MAXNOOFBOARDS:
                i = ddss.MAXNOOFBOARDS
                while i < len(hands_pbn):
                    more = self._solve_helper(strain_i, leader_i, current_trick,
                                              hands_pbn[i:i + ddss.MAXNOOFBOARDS], solutions)
                    for card, values in more.items():
                        results[card] = results[card] + values
                    i += ddss.MAXNOOFBOARDS
        return results

    # ------------------------------------------------------------------
    # Utility methods (pure Python, no library dependency)
    # ------------------------------------------------------------------

    def expected_tricks_dds(self, card_results):
        return {card: round(sum(v) / len(v), 2) for card, v in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list: List[float]):
        probs = [float(p) for p in probabilities_list]
        return {card: round(sum(p * r for p, r in zip(probs, rl)), 2)
                for card, rl in card_results.items()}

    def p_made_target(self, tricks_needed):
        def fun(card_results):
            return {card: round(sum(1 for x in v if x >= tricks_needed) / len(v), 3)
                    for card, v in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}"
            f"{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        sorted_counts_dict = {}
        for key, array in dd_solved.items():
            element_count = Counter(array)
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)
            sorted_counts_dict[key] = sorted_counts

        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")

    # ------------------------------------------------------------------
    # DDSS-only: batch par calculation using CalcAllTablesPBNx
    # ------------------------------------------------------------------

    def calculatepar_batch(self, hands, vulns):
        """Calculate par for multiple hands at once using CalcAllTablesPBNx.

        Args:
            hands: list of PBN hand strings (without "N:" prefix)
            vulns: list of (ns_vul, ew_vul) boolean tuples

        Returns:
            list of NS par scores (int), or None for failed hands
        """
        if self._fallback:
            # Fallback: call single par calculation in a loop
            return [self._fallback.calculatepar(h, v, print_result=False)
                    for h, v in zip(hands, vulns)]

        n = len(hands)
        DealArray = ddss.ddTableDealPBN * n
        ResultArray = ddss.ddTableResults * n
        ParArray = ddss.parResults * n

        deals = DealArray()
        results = ResultArray()
        par_out = ParArray()

        for i, hand in enumerate(hands):
            deals[i].cards = ("N:" + hand).encode('utf-8')

        # mode = -1 means no par calculation inside CalcAllTablesPBNx
        # We'll do par separately per hand to handle vulnerability correctly
        trump_filter = (ctypes.c_int * ddss.DDS_STRAINS)(0, 0, 0, 0, 0)

        res = ddss.CalcAllTablesPBNx(n, deals, -1, trump_filter, results, None)
        if res != 1:
            msg = ddss.get_error_message(res)
            sys.stderr.write(f"{Fore.RED}CalcAllTablesPBNx error: {res} — {msg}{Style.RESET_ALL}\n")
            return [None] * n

        # Now calculate par per hand (Par is fast, the DD table was the expensive part)
        scores = []
        for i in range(n):
            vuln = vulns[i]
            v = 0
            if vuln[0]: v = 2
            if vuln[1]: v = 3
            if vuln[0] and vuln[1]: v = 1

            pres = ddss.parResults()
            par_res = ddss.Par(ctypes.pointer(results[i]), pres, v)
            if par_res != 1:
                scores.append(None)
                continue

            par_str = pres.parScore[0].value.decode('utf-8')
            ns_score = par_str.split()[1]
            scores.append(int(ns_score))

        return scores

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _get_ddss_version(self):
        """Query version from the DDSS library via GetDDSInfo."""
        try:
            info = ddss.DDSInfo()
            ddss.GetDDSInfo(ctypes.pointer(info))
            return info.versionString.decode('utf-8')
        except Exception:
            return "2.9.0-ddss"

    def _calculatepar_impl(self, hand, vuln, print_result=True):
        tableDealPBN = ddss.ddTableDealPBN()
        table = ddss.ddTableResults()
        myTable = ctypes.pointer(table)

        tableDealPBN.cards = ("N:" + hand).encode('utf-8')

        res = ddss.CalcDDtablePBN(tableDealPBN, myTable)
        if res != 1:
            error_message = ddss.get_error_message(res)
            sys.stderr.write(f"Error Code: {res}, Error Message: {error_message}, Hand {hand.encode('utf-8')}\n")
            raise Exception(error_message)

        pres = ddss.parResults()

        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        res = ddss.Par(myTable, pres, v)
        if res != 1:
            error_message = ddss.get_error_message(res)
            sys.stderr.write(f"{Fore.RED}Error Code: {res}, Error Message: {error_message} {hand.encode('utf-8')}{Style.RESET_ALL}")
            return None

        par = ctypes.pointer(pres)

        if print_result:
            print("NS score: {}".format(par.contents.parScore[0].value.decode('utf-8')))
            print("EW score: {}".format(par.contents.parScore[1].value.decode('utf-8')))

        par_str = par.contents.parScore[0].value.decode('utf-8')
        ns_score = par_str.split()[1]
        return int(ns_score)

    def _solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200,
                     0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        n = len(hands_pbn)
        bo = ddss.AllocBoardsPBN(n)
        solved = ddss.AllocSolvedBoards(n)

        try:
            for handno in range(n):
                bo[0].deals[handno].trump = (strain_i - 1) % 5
                bo[0].deals[handno].first = leader_i

                for i in range(3):
                    bo[0].deals[handno].currentTrickSuit[i] = 0
                    bo[0].deals[handno].currentTrickRank[i] = 0
                    if i < len(current_trick):
                        bo[0].deals[handno].currentTrickSuit[i] = current_trick[i] // 13
                        bo[0].deals[handno].currentTrickRank[i] = 14 - current_trick[i] % 13

                bo[0].deals[handno].remainCards = hands_pbn[handno].encode('utf-8')

                bo[0].target[handno] = -1
                bo[0].solutions[handno] = solutions
                bo[0].mode[handno] = self.dds_mode

            res = ddss.SolveAllBoards(bo, solved)
            if res != 1:
                error_message = ddss.get_error_message(res)
                print(f"{Fore.RED}Error Code: {res}, Error Message: {error_message} "
                      f"{hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
                return None

            if solutions == 1:
                card_results = {}
                card_results["max"] = []
                card_results["min"] = []
                for handno in range(n):
                    fut = solved[0].solvedBoard[handno]
                    suit_i = fut.suit[0]
                    card = suit_i * 13 + 14 - fut.rank[0]
                    card_results["max"].append(fut.score[0])
                    card_results["min"].append(fut.score[fut.cards - 1])
            else:
                card_results = {}
                for handno in range(n):
                    fut = solved[0].solvedBoard[handno]
                    for i in range(fut.cards):
                        suit_i = fut.suit[i]
                        card = suit_i * 13 + 14 - fut.rank[i]
                        if card not in card_results:
                            card_results[card] = []
                        card_results[card].append(fut.score[i])
                        eq_cards_encoded = fut.equals[i]
                        for k, rank_code in enumerate(card_rank):
                            if rank_code & eq_cards_encoded > 0:
                                eq_card = suit_i * 13 + k
                                if eq_card not in card_results:
                                    card_results[eq_card] = []
                                card_results[eq_card].append(fut.score[i])

            return card_results

        finally:
            ddss.FreeBoardsPBN(bo)
            ddss.FreeSolvedBoards(solved)
