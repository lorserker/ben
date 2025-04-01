import re

def validate_lead(hand, lead, contract_type, verbose):
    """
    Validates if a lead is correct according to bridge opening lead rules.
    
    Args:
        hand (str): The cards in the hand for a specific suit (e.g., "QJTxx")
        lead (str): The card led (e.g., "Q")
        contract_type (str): "suit" for suit contracts, "nt" for no-trump contracts
    
    Returns:
        bool: True if the lead is correct, False otherwise
        str: Explanation of the correct lead
    """
    contract_type = contract_type.lower()
    
    if contract_type not in ["suit", "nt"]:
        return False, "Contract type must be 'suit' or 'nt'."
    
    # Check if the lead card is in the hand
    if lead not in hand:
        return False, f"The lead card {lead} is not in the hand {hand}."
    
    # Define patterns and their correct leads for suit contracts
    suit_rules = [
        # Axxx... patterns
        (r"^A[^KQJ][^KQJ][^KQJ].*$", "A", "Leading from Axxx... vs suit"),
        (r"^A[^KQJ][^KQJ][^KQJ].*$", "x", "Leading from Axxx... vs suit"),
        (r"^AQ[^J].*$", "A", "Leading from AQxxx... vs suit"),
        # Hxxx... patterns (where H is A, K, Q, J, or T)
        (r"^[AKQJ][^AKQJT][^AKQJT][^AKQJT].*$", "x", "Leading from Hxxx... vs suit"),
        # Hxx patterns
        (r"^[AKQJ][^AKQJ][^AKQJ]$", "x", "Leading from Hxx vs suit"),
        
        # Small card patterns
        (r"^[^AKQJT][^AKQJT][^AKQJT][^AKQJT].*$", "x", "Leading from small cards (xxxx...) vs suit"),
        (r"^[^AKQJT][^AKQJT][^AKQJT]$", "x", "Leading from small cards (xxx) vs suit"),
        (r"^T[^AKQJT][^AKQJT].*$", "T", "Leading from Txx... vs suit"),
        (r"^T[^AKQJT][^AKQJT].*$", "x", "Leading from Txx... vs suit"),
        
        # Honor combinations
        (r"^[AKQJT][^AKQJT]$", hand[0], "Leading from Hx vs suit"),
        (r"^[^AKQJT][^AKQJT]$", "x", "Leading from xx vs suit"),
        (r"^AK.*$", "A", "Leading from AK sequence vs suit"),
        (r"^KQ.*$", "K", "Leading from KQ sequence vs suit"),
        (r"^QJ.*$", "Q", "Leading from QJ sequence vs suit"),
        
        # Sequences with T
        (r"^AKQJ?T.*$", "A", "Leading from AKQ(J)T sequence vs suit"),
        (r"^KQJT.*$", "K", "Leading from KQJT sequence vs suit"),
        (r"^QJT[^AKQJ].*$", "Q", "Leading from QJTx sequence vs suit"),
        (r"^JT9[^AKQJ].*$", "J", "Leading from JT9x sequence vs suit"),
        
        # Broken sequences
        (r"^AKJ[^Q].*$", "A", "Leading from AKJx sequence vs suit"),
        (r"^KQT[^J].*$", "K", "Leading from KQTx sequence vs suit"),
        (r"^QJ9[^T].*$", "Q", "Leading from QJ9x sequence vs suit"),
        (r"^AK[^QJ].*$", "A", "Leading from AKxx sequence vs suit"),
        (r"^KQ[^JT].*$", "K", "Leading from KQxx sequence vs suit"),
        (r"^QJ[^T9].*$", "Q", "Leading from QJxx sequence vs suit"),
        
        # Additional sequences
        (r"^AJT[^KQ].*$", "A", "Leading from AJTx sequence vs suit"),
        (r"^AJT[^KQ].*$", "J", "Leading from AJTx sequence vs suit"),
        (r"^KJT[^AQ].*$", "K", "Leading from KJTx sequence vs suit"),
        (r"^KJT[^AQ].*$", "J", "Leading from KJTx sequence vs suit"),
        (r"^QT[^AKJ].*$", "Q", "Leading from QT9x sequence vs suit"),
        (r"^QT[^AKJ].*$", "T", "Leading from QT9x sequence vs suit"),
    ]
    
    # Define patterns for no-trump contracts
    nt_rules = [
        # Similar patterns as suit rules but with potentially different leads for NT
        (r"^A[^K][^KQ][^KQJ].*$", "A", "Leading from Axxx... vs NT"),
        (r"^A[^K][^KQ][^KQJ].*$", "x", "Leading from Axxx... vs NT"),
        (r"^[AKQJT][^AKQJT][^AKQJT][^AKQJT].*$", "x", "Leading from Hxxx... vs NT"),
        (r"^[AKQJT][^AKQJT][^AKQJT]$", "x", "Leading from Hxx vs NT"),
        
        (r"^[^AKQJT][^AKQJT][^AKQJT][^AKQJT].*$", "x", "Leading from small cards (xxxx...) vs NT"),
        (r"^[^AKQJT][^AKQJT][^AKQJT]$", "x", "Leading from small cards (xxx) vs NT"),
        (r"^T[^AKQJT][^AKQJT].*$", "T", "Leading from Txx... vs NT"),
        (r"^T[^AKQJT][^AKQJT].*$", "x", "Leading from Txx... vs NT"),
        
        (r"^[AKQJT][^AKQJT]$", hand[0], "Leading from Hx vs NT"),
        (r"^[^AKQJT][^AKQJT]$", "x", "Leading from xx vs NT"),
        (r"^AK.*$", "A", "Leading from AK sequence vs NT"),
        (r"^KQ.*$", "K", "Leading from KQ sequence vs NT"),
        (r"^QJ.*$", "Q", "Leading from QJ sequence vs NT"),
        
        (r"^AKQJ?T.*$", "A", "Leading from AKQ(J)T sequence vs NT"),
        (r"^KQJT.*$", "K", "Leading from KQJT sequence vs NT"),
        (r"^QJT[^AKQJ].*$", "Q", "Leading from QJTx sequence vs NT"),
        (r"^JT9[^AKQJ].*$", "J", "Leading from JT9x sequence vs NT"),
        
        (r"^AKJ[^Q].*$", "A", "Leading from AKJx sequence vs NT"),
        (r"^KQT[^J].*$", "K", "Leading from KQTx sequence vs NT"),
        (r"^QJ9[^T].*$", "Q", "Leading from QJ9x sequence vs NT"),
        
        (r"^AK[^QJ].*$", "A", "Leading from AKxx sequence vs NT"),
        (r"^KQ[^JT].*$", "K", "Leading from KQxx sequence vs NT"),
        (r"^QJ[^T9].*$", "Q", "Leading from QJxx sequence vs NT"),
        
        (r"^AJT[^KQ].*$", "A", "Leading from AJTx sequence vs NT"),
        (r"^AJT[^KQ].*$", "J", "Leading from AJTx sequence vs NT"),
        (r"^KJT[^AQ].*$", "K", "Leading from KJTx sequence vs NT"),
        (r"^KJT[^AQ].*$", "j", "Leading from KJTx sequence vs NT"),
        (r"^QT9[^AKJ].*$", "Q", "Leading from QT9x sequence vs NT"),
        (r"^QT9[^AKJ].*$", "T", "Leading from QT9x sequence vs NT"),
        
        # NT-specific rule
        (r"^AKJT.*$", "A", "Leading from AKJT sequence vs NT"),
        (r"^AKJT.*$", "K", "Leading from AKJT sequence vs NT")
    ]
    
    # Choose the appropriate set of rules
    rules = suit_rules if contract_type == "suit" else nt_rules
    
    # Track if any rule allows the lead
    valid_explanations = []
    
    for pattern, correct_lead, explanation in rules:
        if re.match(pattern, hand):
            if lead == correct_lead:
                return True, f"Valid lead: {explanation}. Correct lead is {correct_lead}."
            else:
                valid_explanations.append(f"Rule: {explanation}. Correct lead should be {correct_lead}.")

    if valid_explanations:
        return False, valid_explanations
        
    return True, [f"No matching rule found for hand {hand} vs {contract_type}. Unable to validate lead."]
