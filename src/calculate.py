import scoring

def calculate_mp_score_probability( data, probabilities_list):
    scores = {key: 0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_arrays = len(keys)
    num_plays = len(data[keys[0]])

    if num_arrays == 1:
        scores[keys[0]] = 100
        return scores
    # Compare each array with all others
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(len(data[keys[i]])):  # Iterate through each value (column)
                    if data[keys[i]][k] > data[keys[j]][k]:
                        scores[keys[i]] += 1 * probabilities_list[k] * num_plays
                    elif data[keys[i]][k] < data[keys[j]][k]:
                        scores[keys[i]] -= 1 * probabilities_list[k] * num_plays
    max_mp_score = 2 * num_plays * (num_arrays - 1)
    #print(max_mp_score, scores)
    #print(probabilities_list)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round(100 * (scores[key] + max_mp_score / 2) / max_mp_score)
    return scores

def calculate_mp_score( data):
    scores = {key: 0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_arrays = len(keys)
    num_plays = len(data[keys[0]])

    if num_arrays == 1:
        scores[keys[0]] = 100
        return scores

    # Compare each array with all others
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(len(data[keys[i]])):  # Iterate through each value (column)
                    if data[keys[i]][k] > data[keys[j]][k]:
                        scores[keys[i]] += 1
                    elif data[keys[i]][k] < data[keys[j]][k]:
                        scores[keys[i]] -= 1
    max_mp_score = 2 * num_plays * (num_arrays - 1)
    #print(max_mp_score, scores)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round(100 * (scores[key] + max_mp_score / 2) / max_mp_score)
    return scores

def calculate_imp_score_probability( data, probabilities_list):
    scores = {key: 0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_arrays = len(keys)
    num_plays = len(data[keys[0]])

    if num_arrays == 1:
        scores[keys[0]] = 0
        return scores

    # Compare each array with all others
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(num_plays):  # Iterate through each value (column)
                    diff = data[keys[i]][k] - data[keys[j]][k]
                    imp_score = scoring.diff_to_imps(diff) * probabilities_list[k] * num_plays
                    #if i == 0 and j == 5:
                    #    print(diff, imp_score, probabilities_list[k])
                    # Add or subtract the IMP score based on the sign of diff
                    if diff >= 0:
                        scores[keys[i]] += imp_score
                    else:
                        scores[keys[i]] -= imp_score        

    num_scores = num_plays  * (num_arrays - 1)
    #print(num_score, scores)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round((scores[key]) / num_scores, 2) 
    return scores

def calculate_imp_score( data):
    scores = {key: 0 for key in data}  # Initialize scores for each key
    keys = list(data.keys())  # Get the list of keys
    num_arrays = len(keys)
    num_plays = len(data[keys[0]])

    if num_arrays == 1:
        scores[keys[0]] = 0
        return scores

    # Compare each array with all others
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                # Compare data[keys[i]] with data[keys[j]] column by column
                for k in range(num_plays):  # Iterate through each value (column)
                    diff = data[keys[i]][k] - data[keys[j]][k]
                    imp_score = scoring.diff_to_imps(diff)
                    
                    # Add or subtract the IMP score based on the sign of diff
                    if diff >= 0:
                        scores[keys[i]] += imp_score
                    else:
                        scores[keys[i]] -= imp_score        
    num_scores = num_plays * (num_arrays - 1)
    #print(num_scores, scores)
    # Translate the scores to percentages
    for key in scores:
        scores[key] = round((scores[key]) / num_scores, 2) 
    return scores

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
    for card, future_tricks in dd_solved.items():
        ev_sum = 0
        for ft, proba in zip(future_tricks, probabilities_list):
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
    for card, future_tricks in dd_solved.items():
        ev_sum = 0
        for ft, proba in zip(future_tricks, probabilities_list):
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
