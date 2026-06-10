import sys
import scoring

def check_array_lengths(dictionary):
    lengths = [len(value) for value in dictionary.values()]
    return min(lengths)


def calculate_mp_score_probability( data, probabilities_list):
    scores = {key: 0.0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_arrays = len(keys)
    num_plays = check_array_lengths(data)
    # Convert to plain Python lists to avoid numpy scalar overhead in tight loops
    probs = [float(p) for p in probabilities_list[:num_plays]]
    data_lists = {k: [float(v) for v in vals[:num_plays]] for k, vals in data.items()}

    if num_arrays == 1:
        scores[keys[0]] = 100
        return scores
    # Compare each array with all others
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                di = data_lists[keys[i]]
                dj = data_lists[keys[j]]
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(num_plays):  # Iterate through each value (column)
                    if di[k] > dj[k]:
                        scores[keys[i]] += probs[k] * num_plays
                    elif di[k] < dj[k]:
                        scores[keys[i]] -= probs[k] * num_plays
    max_mp_score = 2 * num_plays * (num_arrays - 1)
    #print(max_mp_score, scores)
    #print(probabilities_list)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round(100 * (scores[key] + max_mp_score / 2) / max_mp_score)
    return scores

def _round_robin_score(data, value_fn, weights=None):
    """Round-robin comparison of each candidate against every other candidate.

    This is the single shared method behind both matchpoint and IMP scoring; the
    only thing that differs is ``value_fn`` (the per-comparison value) and hence
    the output scale:

      * matchpoints -> value_fn = sign(diff)          -> result scale  -1 .. +1
      * IMPs        -> value_fn = signed diff_to_imps  -> result scale -24 .. +24

    Contract (must match Brill's DdsResultAnalyzer / EndgamePlayEngine):

      data    : {key: [score_per_sample]}. Every list MUST have the same length
                and be aligned by sample index -- sample ``k`` is the SAME world
                (deal) for every key. Do NOT sort the lists independently and do
                NOT collapse them to per-key histograms; either destroys the
                pairing and silently corrupts the result.
      value_fn: maps a signed score difference (score_i - score_j) to a signed
                contribution.
      weights : optional per-sample weight list, parallel to the score lists.
                ``None`` weights every sample 1.0. Weighting by a constant gives
                identical results to no weights.

    Returns {key: average signed value against the field}.
    """
    keys = list(data.keys())
    n = len(keys)
    if n == 0:
        return {}
    if n == 1:
        # Nothing to compare against -> neutral on either scale.
        return {keys[0]: 0.0}

    num_samples = len(data[keys[0]])
    for key in keys:
        if len(data[key]) != num_samples:
            raise ValueError(
                f"All score lists must have the same length and be aligned by "
                f"sample/world; '{key}' has {len(data[key])}, expected "
                f"{num_samples}. (An upstream SortResults() or histogram "
                f"aggregation likely broke the per-world alignment.)"
            )

    if weights is None:
        weights = [1.0] * num_samples
    elif len(weights) != num_samples:
        raise ValueError(
            f"weights length ({len(weights)}) must match the number of "
            f"samples ({num_samples})."
        )

    # Plain Python floats to avoid numpy scalar overhead in the hot loop.
    cols = {key: [float(v) for v in data[key]] for key in keys}
    w = [float(x) for x in weights]
    total_weight = sum(w)
    if total_weight <= 0:
        return {key: 0.0 for key in keys}

    norm = total_weight * (n - 1)
    scores = {}
    for i in range(n):
        di = cols[keys[i]]
        acc = 0.0
        for j in range(n):
            if i == j:
                continue
            dj = cols[keys[j]]
            for k in range(num_samples):
                acc += value_fn(di[k] - dj[k]) * w[k]
        scores[keys[i]] = acc / norm
    return scores


def _mp_value(diff):
    """Matchpoint contribution for one paired comparison: win / tie / loss."""
    if diff > 0:
        return 1.0
    if diff < 0:
        return -1.0
    return 0.0


def _imp_value(diff):
    """Signed IMP contribution for one paired comparison.

    scoring.diff_to_imps() returns an UNSIGNED bucket, so apply the sign here.
    """
    imps = scoring.diff_to_imps(diff)
    return imps if diff >= 0 else -imps


def calculate_mp_score(data, weights=None):
    """Matchpoint score per candidate on a -1 .. +1 scale.

    +1 = beats the field on every (weighted) sample, -1 = loses to the field on
    every sample, 0 = average. Paired per sample and weight-aware; see
    _round_robin_score for the input contract.

    NOTE: this replaces the old 0..100 percentage scale. The previous version
    also (a) ignored sample weights, (b) truncated all lists to the shortest,
    and (c) compared independently-sorted lists -- all of which corrupted the
    result for cards with a wide/rare trick spread. Convert to the old 0..100
    percentage with ``50 * (mp + 1)`` if a consumer still needs one.
    """
    return {key: round(v, 4)
            for key, v in _round_robin_score(data, _mp_value, weights).items()}

def calculate_imp_score_probability( data, probabilities_list):
    scores = {key: 0.0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_plays = len(keys)
    num_samples = check_array_lengths(data)
    # Convert to plain Python lists to avoid numpy scalar overhead in tight loops
    probs = [float(p) for p in probabilities_list[:num_samples]]
    data_lists = {k: [float(v) for v in vals[:num_samples]] for k, vals in data.items()}

    if num_plays == 1:
        scores[keys[0]] = 0
        return scores

    # Compare each array with all others
    for i in range(num_plays):
        for j in range(num_plays):
            if i != j:
                di = data_lists[keys[i]]
                dj = data_lists[keys[j]]
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(num_samples):  # Iterate through each value (column)
                    diff = di[k] - dj[k]
                    imp_score = scoring.diff_to_imps(diff) * probs[k] * num_samples
                    # Add or subtract the IMP score based on the sign of diff
                    if diff >= 0:
                        scores[keys[i]] += imp_score
                    else:
                        scores[keys[i]] -= imp_score

    num_scores = num_samples  * (num_plays - 1)
    #print(num_score, scores)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round((scores[key]) / num_scores, 2) 
    return scores

def calculate_imp_score(data, weights=None):
    """IMP score per candidate on a -24 .. +24 scale (average IMPs vs the field).

    Identical method to calculate_mp_score; only the per-comparison value differs
    (signed diff_to_imps instead of +/-1). Paired per sample and weight-aware;
    see _round_robin_score for the input contract.

    This matches the behaviour of the previous version for equal-length,
    unweighted input -- it adds per-sample weights and now *requires* aligned,
    equal-length lists instead of silently truncating to the shortest.
    """
    return {key: round(v, 2)
            for key, v in _round_robin_score(data, _imp_value, weights).items()}

def calculate_score( dd_solved, n_tricks_taken, player_i, score_by_tricks_taken):
    card_ev = {}
    sign = 1 if player_i % 2 == 1 else -1
    
    for card, future_tricks in dd_solved.items():
        card_ev[card] = []
        
        for ft in future_tricks:
            if ft < 0:
                continue
            tot_tricks = n_tricks_taken + ft
            tot_decl_tricks = tot_tricks if player_i % 2 == 1 else 13 - tot_tricks
            ev = sign * score_by_tricks_taken[tot_decl_tricks]
            
            # Append each individual ev for this card
            card_ev[card].append(round(ev, 2))
    
    return card_ev    

def get_card_ev( dd_solved, n_tricks_taken, player_i, score_by_tricks_taken):
    card_ev = {}
    sign = 1 if player_i % 2 == 1 else -1
    for card, future_tricks in dd_solved.items():
        ev_sum = 0
        for ft in future_tricks:
            if ft < 0:
                continue
            tot_tricks = n_tricks_taken + ft
            tot_decl_tricks = tot_tricks if player_i % 2 == 1 else 13 - tot_tricks
            ev_sum += sign * score_by_tricks_taken[tot_decl_tricks]
        card_ev[card] = ev_sum / len(future_tricks)
            
    for key in card_ev:
        card_ev[key] = round(card_ev[key],2)
    return card_ev

def get_card_ev_probability( dd_solved, probabilities_list, n_tricks_taken, player_i, score_by_tricks_taken):
    card_ev = {}
    sign = 1 if player_i % 2 == 1 else -1
    # Convert to plain Python list to avoid numpy scalar overhead
    probs = [float(p) for p in probabilities_list]
    for card, future_tricks in dd_solved.items():
        ev_sum = 0.0
        for ft, proba in zip(future_tricks, probs):
            if ft < 0:
                continue
            tot_tricks = n_tricks_taken + ft
            tot_decl_tricks = (
                tot_tricks if player_i % 2 == 1 else 13 - tot_tricks
            )
            ev_sum += sign * score_by_tricks_taken[tot_decl_tricks] * proba
        card_ev[card] = ev_sum

    for key in card_ev:
        card_ev[key] = round(card_ev[key],2)
    return card_ev

def get_card_ev_mp_probability( dd_solved, probabilities_list):
    card_ev = {}
    # Convert to plain Python list to avoid numpy scalar overhead
    probs = [float(p) for p in probabilities_list]
    for card, future_tricks in dd_solved.items():
        ev_sum = 0.0
        for ft, proba in zip(future_tricks, probs):
            if ft < 0:
                continue
            ev_sum += ft * proba * 100
        card_ev[card] = ev_sum
    for key in card_ev:
        card_ev[key] = round(card_ev[key],2)
    return card_ev

def get_card_ev_mp( dd_solved, n_tricks_taken, player_i):
    card_ev = {}
    sign = 1 if player_i % 2 == 1 else -1
    for card, future_tricks in dd_solved.items():
        ev_sum = 0
        for ft in future_tricks:
            if ft < 0:
                continue
            tot_tricks = n_tricks_taken + ft
            tot_decl_tricks = tot_tricks if player_i % 2 == 1 else 13 - tot_tricks
            ev_sum += sign * tot_decl_tricks * 100
        card_ev[card] = ev_sum / len(future_tricks)
            
    for key in card_ev:
        card_ev[key] = round(card_ev[key],2)
    return card_ev
