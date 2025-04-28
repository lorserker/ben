from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class Rule:
    pattern: str
    correct_lead: str
    explanation: str
    length_required: Optional[int] = None  # Optional: can be None
    max_length: Optional[int] = None  # Optional: can be None
    lambda_fn: Optional[callable] = None  # Lambda function to dynamically determine lead, if required

def validate_lead(suit: str, lead: str, contract_type: str, verbose):
    """
    Validates if a lead is correct according to bridge opening lead rules.
    
    Args:
        suit (str): The cards in the suit (e.g., "QJTxx" or "7 5")
        lead (str): The card led (e.g., "Q")
        contract_type (str): "suit" for suit contracts, "nt" for no-trump contracts
    
    Returns:
        bool: True if the lead is correct, False otherwise
        list[str]: Explanations of the correct leads
    """
    contract_type = contract_type.lower()
    
    if contract_type not in ["suit", "nt"]:
        return False, ["Contract type must be 'suit' or 'nt'."]
    
    # Clean up suit for actual cards (e.g., "7 5" instead of "75")
    suit = "".join(suit.split())  # Remove any spaces, so we work with the raw string

    if lead not in suit:
        return False, [f"The lead card {lead} is not in the suit {suit}."]

    # This must be 
    # Define the rules using the Rule dataclass
    suit_rules = [
        Rule(r"^AKQ", "A", "Leading from AKQ sequence vs suit"),
        Rule(r"^AKQ", "K", "Leading from AKQ sequence vs suit"),
        Rule(r"^AKQ", "Q", "Leading from AKQ sequence vs suit"),
        Rule(r"^KQJ", "K", "Leading from KQJ sequence vs suit"),
        Rule(r"^QJT", "Q", "Leading from QJT sequence vs suit"),
        Rule(r"^JT9", "J", "Leading from JT9 sequence vs suit"),
        Rule(r"^AKJ", "A", "Leading from AKJ sequence vs suit"),
        Rule(r"^AKJ", "K", "Leading from AKJ sequence vs suit"),
        Rule(r"^KQT", "K", "Leading from KQT sequence vs suit"),
        Rule(r"^QJ9", "Q", "Leading from QJ9 sequence vs suit"),
        Rule(r"^AK", "A", "Leading from AK doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^AK", "K", "Leading from AK doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^KQ", "K", "Leading from KQ doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^QJ", "Q", "Leading from QJ doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^JT", "J", "Leading from JT doubleton vs suit", length_required=2, max_length=2),
        
        # Txx holding - now using lambda for leading T or x
        Rule(r"^T[^AKQJT][^AKQJT]$", "T", "Leading top from Txx vs suit", lambda_fn=lambda suit: 'T'),
        Rule(r"^T[^AKQJT][^AKQJT]$", "x", "Leading small from Txx vs suit", lambda_fn=lambda suit: [card for card in suit if card != 'T']),
        
        # Hxx holdings
        Rule(r"^[AKQJ][^AKQJT][^AKQJT]$", "x", "Leading small from Hxx vs suit", length_required=3, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        Rule(r"^A[^AKQJT]{2}$", "A", "Leading Ace from Axx", length_required=2, lambda_fn=lambda suit: 'A'),  # Ace is always the valid lead
      
        # Jxxx case: leading lowest
        Rule(r"^J[^AKQJT]{3}$", "x", "Leading low from Jxxx vs suit", length_required=4, lambda_fn=lambda suit: [card for card in suit if card != 'J']),
        
        # Small cards
        Rule(r"^[^AKQJT]{3,}$", "x", "Leading low from xxx or longer small cards vs suit", length_required=3, max_length=3, lambda_fn=lambda suit: suit),
        #Rule(r"^[^AKQJT]{2}$", "x", "Leading low from xx vs suit", length_required=2, lambda_fn=lambda suit: 'x'),
        
        # Single honors
        Rule(r"^[AKQJT][^AKQJT]+$", "x", "Leading from Hx vs suit", length_required=2, max_length=2, lambda_fn=lambda suit: suit[0]),
        
        # Rules for actual cards like "7 5"
        Rule(r"[^AKQJT]$", "x", "Leading low from two cards with 9 or below vs suit", lambda_fn=lambda suit: 'x'),
        Rule(r"[^AKQJT]$", "highest", "Leading highest from two cards (lowest is OK for 9 or below) vs suit", lambda_fn=lambda suit: suit[0]),

        Rule(r"^[AKQJT23456789]{1}$",  # Matches any single card from the deck (A, K, Q, J, T, 2-9)
            "any",  # Any card is valid
            "Leading from a single card", 
            length_required=1,  # The suit has only 1 card
            max_length=1,  # Maximum length of the suit is 1
            lambda_fn=lambda suit: suit  # Any card in the suit is valid (return the suit as it is)
        )
    ]
    
    nt_rules = [
        Rule(r"^AKQ", "A", "Leading from AKQ sequence vs NT"),
        Rule(r"^AKQ", "K", "Leading from AKQ sequence vs NT"),
        Rule(r"^AKQ", "Q", "Leading from AKQ sequence vs NT"),
        Rule(r"^KQJ", "K", "Leading from KQJ sequence vs NT"),
        Rule(r"^QJT", "Q", "Leading from QJT sequence vs NT"),
        Rule(r"^JT9", "J", "Leading from JT9 sequence vs NT"),
        Rule(r"^AKJ", "A", "Leading from AKJ sequence vs NT"),
        Rule(r"^AKJ", "K", "Leading from AKJ sequence vs NT"),
        Rule(r"^KQT", "K", "Leading from KQT sequence vs NT"),
        Rule(r"^QJ9", "Q", "Leading from QJ9 sequence vs NT"),
        Rule(r"^AK", "K", "Leading from AK doubleton vs NT", length_required=2,max_length=2),
        Rule(r"^KQ", "K", "Leading from KQ doubleton vs NT", length_required=2,max_length=2),
        Rule(r"^QJ", "Q", "Leading from QJ doubleton vs NT", length_required=2,max_length=2),
        Rule(r"^JT", "J", "Leading from JT doubleton vs NT", length_required=2,max_length=2),
        
        # Txx holding - now using lambda for leading T or x
        Rule(r"^T[^AKQJT][^AKQJT]$", "T", "Leading top from Txx vs NT", lambda_fn=lambda suit: 'T'),
        Rule(r"^T[^AKQJT][^AKQJT]$", "x", "Leading small from Txx vs NT", lambda_fn=lambda suit: [card for card in suit if card != 'T']),
        
        # Hxx holdings
        Rule(r"^[AKQJ][^AKQJT][^AKQJT]$", "x", "Leading small from Hxx vs NT", length_required=3, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        
        # Jxxx case: leading lowest
        Rule(r"^J[^AKQJT]{3}$", "x", "Leading low from Jxxx vs NT", length_required=4, lambda_fn=lambda suit: [card for card in suit if card != 'J']),
        
        # Small cards
        Rule(r"^[^AKQJT]{3,}$", "x", "Leading low from xxx or longer small cards vs NT", lambda_fn=lambda suit: suit),
        Rule(r"^[^AKQJT]{2}$", "x", "Leading low from xx vs NT", length_required=2, lambda_fn=lambda suit: suit),
        
        # Single honors
        Rule(r"^[AKQJT][^AKQJT]+$", "highest", "Leading from Hx vs NT", length_required=2, max_length=2, lambda_fn=lambda suit: suit[0]),
        Rule(r"^A[^AKQJT]{2}$", "A", "Leading Ace from Axx", length_required=2, lambda_fn=lambda suit: 'A'),  # Ace is always the valid lead
        
        # Rules for actual cards like "7 5"
        Rule(r"^[2-7][^AKQJT]$", "x", "Leading low from two cards with 9 or below vs NT", lambda_fn=lambda suit: 'x'),
        Rule(r"^[2-7][^AKQJT]$", "highest", "Leading highest from two cards (lowest is OK for 9 or below) vs NT", lambda_fn=lambda suit: 'highest')
    ]
    
    # Pick correct rule set
    rules = suit_rules if contract_type == "suit" else nt_rules

    # Find matching rule
    valid_explanations = []
    for rule in rules:
        if re.match(rule.pattern, suit):
            # Check if length is required
            if rule.length_required and len(suit) < rule.length_required:
                continue  # Skip this rule if the length doesn't match
            if rule.max_length and len(suit) > rule.max_length:
                continue  # Skip this rule if the length doesn't match

            # If lambda function exists, use it to determine the expected lead, else use the static correct lead
            if rule.lambda_fn:
                valid_leads = rule.lambda_fn(suit)  # Use the lambda function to get valid lead cards
                if lead in valid_leads:
                    return True, [f"Valid lead: {rule.explanation}. Correct lead is {lead}."]
                else:
                    valid_explanations.append(f"Rule: {rule.explanation}. Correct lead should be {valid_leads}.")
            else:
                expected_lead = rule.correct_lead
                if expected_lead == lead:                    
                    return True, [f"Valid lead: {rule.explanation}. Correct lead is {expected_lead}."]
                else:
                    valid_explanations.append(f"Rule: {rule.explanation}. Correct lead should be {expected_lead}.")

    if valid_explanations:
        return False, valid_explanations

    return True, [f"No matching rule found for suit {suit} vs {contract_type}."]

