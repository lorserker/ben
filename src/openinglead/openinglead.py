from dataclasses import dataclass
from typing import Optional
import re
from colorama import Fore, Back, Style, init

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
    #print("lead", lead, "suit", suit, "contract_type", contract_type)
    if contract_type not in ["suit", "nt"]:
        return False, ["Contract type must be 'suit' or 'nt'."]
    
    # Clean up suit for actual cards (e.g., "7 5" instead of "75")
    suit = "".join(suit.split())  # Remove any spaces, so we work with the raw string

    # As we might have to match 9 or 8 with x, we do not check
    #if lead not in suit:
    #    return False, [f"The lead card {lead} is not in the suit {suit}."]

    # This must be 
    # Define the rules using the Rule dataclass
    suit_rules = [
        Rule(r"^AKQ([2-9x])", "A", "Leading from AKQ sequence vs suit"),
        Rule(r"^AKQ([2-9x])", "K", "Leading from AKQ sequence vs suit"),
        Rule(r"^AKQ([2-9x])", "Q", "Leading from AKQ sequence vs suit"),
        Rule(
            pattern=r"^AQ$",          # Matches the string "AQ" exactly.
            correct_lead="A",
            explanation="Leading A from AQ doubleton vs suit",
            length_required=2,      # Explicitly require length 2
            max_length=2            # Explicitly require length 2
        ),

        Rule(r"^KQJ([2-9x])", "K", "Leading from KQJ sequence vs suit"),
        Rule(r"^QJT([2-9x])", "Q", "Leading from QJT sequence vs suit"),
        Rule(r"^JT9([2-9x])", "J", "Leading from JT9 sequence vs suit"),
        Rule(r"^AKJ([2-9x])", "A", "Leading from AKJ sequence vs suit"),
        Rule(r"^AKJ([2-9x])", "K", "Leading from AKJ sequence vs suit"),
        Rule(r"^KQT([2-9x])", "K", "Leading from KQT sequence vs suit"),
        Rule(r"^KJT([2-9x])*$", "J", "Leading from KJT sequence vs suit"),
        Rule(r"^KJT([2-9x])*$", "T", "Leading from KJT sequence vs suit"),
        Rule(r"^QJ9", "Q", "Leading from QJ9 sequence vs suit"),
        Rule(r"^AK", "A", "Leading from AK doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^AK", "K", "Leading from AK doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^KQ([^AKQ]){0,}$", "K", "Leading from KQ(x) doubleton vs suit", length_required=2),
        Rule(r"^QJ([^AKQJ])?$", "Q", "Leading from QJ(x) doubleton vs suit", length_required=2),
        Rule(r"^JT", "J", "Leading from JT doubleton vs suit", length_required=2, max_length=2),
        Rule(r"^T9", "T", "Leading from T9 doubleton vs suit", length_required=2, max_length=2),
        
        # Txx holding - now using lambda for leading T or x
        Rule(r"^T[^AKQJT][^AKQJT]$", "T", "Leading top from Txx vs suit", lambda_fn=lambda suit: 'T'),
        Rule(r"^T[^AKQJT][^AKQJT]$", "x", "Leading small from Txx vs suit", lambda_fn=lambda suit: [card for card in suit if card != 'T']),
        
        # Hxx holdings
        Rule(r"^[AKQJ][^AKQ][^AKQ]{0,}$", "x", "Leading small from Hxx vs suit", length_required=3, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        Rule(r"^A[^AKQJT]{2}$", "A", "Leading Ace from Axx", length_required=2, lambda_fn=lambda suit: 'A'),  # Ace is always the valid lead
      
        # Jxxx case: leading lowest
        Rule(r"^J[^AKQJT]{3,}$", "x", "Leading low from Jxxx vs suit", length_required=4, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        # Txxx case: leading lowest
        Rule(r"^T[^AKQJT]{3,}$", "x", "Leading low from Jxxx vs suit", length_required=4, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        
        # Small cards
        Rule(r"^[^AKQJT]{3,}$", "x", "Leading low from xxx or longer small cards vs suit", length_required=3, lambda_fn=lambda suit: [card for card in suit if card in '98765432x']),
        Rule(r"^[^AKQJ]{2}$", "x", "Leading high from xx vs suit", length_required=2, lambda_fn=lambda suit: 'x'),
        
        # Single honors
        Rule(r"^[AKQJT][^AKQ]+$", "x", "Leading from Hx vs suit", length_required=2, max_length=2, lambda_fn=lambda suit: suit[0]),
        
        # Rules for actual cards like "7 5"
        Rule(r"[^AKQJT]$", "x", "Leading low from two cards with 9 or below vs suit", lambda_fn=lambda suit: 'x'),
        Rule(r"[^AKQJT]$", "highest", "Leading highest from two cards vs suit", lambda_fn=lambda suit: suit[0]),

        Rule(r"^[AKQJT23456789x]{1}$",  # Matches any single card from the deck (A, K, Q, J, T, 2-9)
            "any",  # Any card is valid
            "Leading from a single card", 
            length_required=1,  # The suit has only 1 card
            max_length=1,  # Maximum length of the suit is 1
            lambda_fn=lambda suit: suit[0]  # Any card in the suit is valid (return the suit as it is)
        ),
        Rule(
            pattern=r"^AQ([2-9Tx])+$",  # A, Q, then one or more small cards (2-9 or 'x')
            correct_lead="A",
            explanation="Leading A from AQ + one or more smalls (e.g., AQx, AQxx, AQxxx) vs suit",
            length_required=2          # Minimum length is AQx (3 cards)
            # No lambda needed if the lead is always 'A'
        ),
        Rule(
            pattern=r"^AQJ([2-9Tx])*$", # A, Q, J, then zero or more small cards (2-9 or 'x')
            correct_lead="A",
            explanation="Leading A from AQJ + zero or more smalls (e.g., AQJ, AQJx, AQJxx) vs suit",
            length_required=3          # Minimum length is AQJ (3 cards)
        ),
    ]
    
    nt_rules = [
        Rule(r"^AKQ", "A", "Leading from AKQ sequence vs NT"),
        Rule(r"^AKQ", "K", "Leading from AKQ sequence vs NT"),
        Rule(r"^AKQ", "Q", "Leading from AKQ sequence vs NT"),
        Rule(r"^AQ", "A", "Leading from AQ sequence vs NT"),
        Rule(r"^KJ", "K", "Leading from KJ sequence vs NT"),
        Rule(r"^KQJ", "K", "Leading from KQJ sequence vs NT"),
        Rule(r"^QJT", "Q", "Leading from QJT sequence vs NT"),
        Rule(r"^JT9", "J", "Leading from JT9 sequence vs NT"),
        Rule(r"^AKJ", "A", "Leading from AKJ sequence vs NT"),
        Rule(r"^AKJ", "K", "Leading from AKJ sequence vs NT"),
        Rule(r"^KQT", "K", "Leading from KQT sequence vs NT"),
        Rule(r"^QJ9", "Q", "Leading from QJ9 sequence vs NT"),
        Rule(r"^JT8", "J", "Leading from QJ9 sequence vs NT"),
        Rule(r"^KJT([2-9x])*$", "J", "Leading from KJT sequence vs nt"),
        Rule(r"^KJT([2-9x])*$", "T", "Leading from KJT sequence vs nt"),
        Rule(r"^AK", "K", "Leading from AK doubleton vs NT", length_required=2,max_length=2),
            # NEW rule for AJT...
        Rule(
            pattern=r"^AJT([2-9x])*$",
            correct_lead="J",
            explanation="Leading J from AJT... (AJT, AJTx, AJTxx, etc.) vs NT",
            length_required=3
        ),
        Rule(r"^KQ([^AKQ])?$", "K", "Leading from KQ(x) doubleton vs NT", length_required=2),
        Rule(
            pattern=r"^KQ([2-9x]){2,}$", # K, Q, then TWO OR MORE small cards (2-9 or 'x')
            correct_lead="K",           # The lead is still K
            explanation="Leading K from KQ + 2 or more smalls (e.g., KQxx, KQxxx) vs NT",
            length_required=4,          # Minimum length is KQxx (4 cards)
            # No lambda needed as the lead is always K
        ),
        Rule(r"^QJ([^AKQJ])?$", "Q", "Leading from QJ(x) doubleton vs NT", length_required=2),
        Rule(
            pattern=r"^QJ([^AKQJ])*$",  # Q, J, then ZERO OR MORE cards that are not A, K, Q, J.
            correct_lead="Q",          # Still lead Q (standard from QJ, QJx, QJxx, QJxxx, QJT...)
            explanation="Leading Q from QJ... (QJ, QJx, QJxx, QJT, etc.) vs NT",
            length_required=2          # Minimum length is QJ (2 cards)
            # max_length is not strictly needed if the '*' handles variable length,
            # but you can add it if you want to cap the length this rule applies to.
            # For example, max_length=5 if you only want it for up to QJxxx.
        ),
        # Rule 1b: Leading small from QJxx vs SUIT
        Rule(
            pattern=r"^QJ([2-9x]){2}$",  # Same pattern
            correct_lead="x",            # Indicates a small card is expected
            explanation="Leading small from QJxx (QJ + two smalls, no Ten) vs suit",
            length_required=4,
            # Lambda to specify which small cards are valid.
            # This returns both small cards as options. Your validation logic handles 'x' vs actual digit.
            lambda_fn=lambda suit_str: [card for card in suit_str[2:] if card in '98765432x']
            # If you specifically mean 4th best (which would be the lower of the two 'x's):
            # lambda_fn=lambda s: [sorted([c for c in s[2:] if c in '98765432x'])[0]] if len([c for c in s[2:] if c in '98765432x']) == 2 else []
        ),        
        Rule(r"^JT", "J", "Leading from JT doubleton vs NT", length_required=2,max_length=2),
        Rule(r"^T9", "T", "Leading from T9 vs NT"),
        
        # Txx holding - now using lambda for leading T or x
        Rule(r"^T[^AKQJT][^AKQJT]$", "T", "Leading top from Txx vs NT", lambda_fn=lambda suit: 'T'),
            # Specific T9x rule
        Rule(
            pattern=r"^T9([^AKQJT9])$", # T, then 9, then a small card not 9
            correct_lead="9",
            explanation="Leading 9 from T9x vs NT",
            length_required=3,
            max_length=3
            # No lambda needed if correct_lead is static '9'
        ),

        # Specific T8x rule (comes after T9x)
        Rule(
            pattern=r"^T8([^AKQJT98])$", # T, then 8, then a small card not 9 or 8
            correct_lead="8",
            explanation="Leading 8 from T8x vs NT",
            length_required=3,
            max_length=3
            # No lambda needed if correct_lead is static '8'
        ),
        Rule(
            pattern=r"^T([2-7x])([2-7x])$", # T followed by two cards from '2'-'7' or 'x'
                                            # (Assuming 'x' represents something < 8)
                                            # Or more broadly: ^T([^AKQJT98])([^AKQJT98])$
            correct_lead="", # Lambda will specify
            explanation="Leading T or small from general Txx (e.g., T7x, Txx with x < 8) vs NT",
            length_required=3,
            max_length=3,
            lambda_fn=lambda suit_str: list(dict.fromkeys(['T', suit_str[1], suit_str[2]]))
        ),        
        # Hxx holdings
        Rule(r"^[AKQJ][^AKQJT]{2,}$", "x", "Leading small from Hxx vs NT", length_required=3, lambda_fn=lambda suit: [card for card in suit if card in 'JT98765432x']),
        
        # Jxxx case: leading lowest
        Rule(r"^J[^AKQJT]{3}$", "x", "Leading low from Jxxx vs NT", length_required=4, lambda_fn=lambda suit: [card for card in suit if card != 'J']),
        
        # Small cards
        Rule(r"^[^AKQJT]{3,}$", "x", "Leading low from xxx or longer small cards vs NT", lambda_fn=lambda suit: [card for card in suit if card in 'JT98765432x']),
        Rule(r"^[^AKQJT]{2}$", "x", "Leading low from xx vs NT", length_required=2, lambda_fn=lambda suit: suit),
        
        # Single honors
        Rule(r"^[AKQJT][^AKQ]+$", "highest", "Leading from Hx vs NT", length_required=2, max_length=2, lambda_fn=lambda suit: suit[0]),
        Rule(r"^A[^AKQJT]{2}$", "A", "Leading Ace from Axx", length_required=2, lambda_fn=lambda suit: 'A'),  # Ace is always the valid lead
        
        # Rules for actual cards like "7 5"
        Rule(r"^[2-7][^AKQJT]$", "x", "Leading low from two cards with 9 or below vs NT", lambda_fn=lambda suit: 'x'),
        Rule(r"^[2-7][^AKQJT]$", "highest", "Leading highest from two cards (lowest is OK for 9 or below) vs NT", lambda_fn=lambda suit: 'highest'),
        Rule(
            pattern=r"^KJ([2-9x]){2}$", # K, J, then two cards from [2-9x]
                                        # Assumes KJT rule is earlier.
            correct_lead="x",           # Example: Lead small from KJ + two smalls
            explanation="Leading small from KJ + two smalls (e.g., KJ98, KJxx, but not KJTx) vs NT",
            length_required=3,
            # Lambda to select the actual small card(s) if 'x' is too generic
            lambda_fn=lambda suit_str: [card for card in suit_str[2:] if card in '98765432x']
                                        # e.g., from KJ98 -> returns ['9','8']
                                        # from KJxx -> returns ['x','x']
        ),
        Rule(
            pattern=r"^AQ([2-9Tx])+$",  # A, Q, then one or more small cards (2-9 or 'x')
            correct_lead="A",
            explanation="Leading A from AQ + one or more smalls (e.g., AQx, AQxx, AQxxx) vs suit",
            length_required=2          # Minimum length is AQx (3 cards)
            # No lambda needed if the lead is always 'A'
        ),
        Rule(
            pattern=r"^AQJ([2-9Tx])*$", # A, Q, J, then zero or more small cards (2-9 or 'x')
            correct_lead="A",
            explanation="Leading A from AQJ + zero or more smalls (e.g., AQJ, AQJx, AQJxx) vs suit",
            length_required=3          # Minimum length is AQJ (3 cards)
        ),
        Rule(r"^[AKQJT23456789]{1}$",  # Matches any single card from the deck (A, K, Q, J, T, 2-9)
            "any",  # Any card is valid
            "Leading from a single card", 
            length_required=1,  # The suit has only 1 card
            max_length=1,  # Maximum length of the suit is 1
            lambda_fn=lambda suit: suit  # Any card in the suit is valid (return the suit as it is)
        ),
        Rule(
            pattern=r"^A([2-9xT])+$",
            correct_lead="x", # Symbolically means "a small card"
            explanation="Leading small from AT + one or more smalls (e.g., ATx, ATxx) vs NT",
            length_required=3,
            # The simplified lambda is usually best here.
            # Your main validation logic handles: if expected_token == 'x' and actual_lead in '2'-'9'.
            lambda_fn=lambda suit_str: [card for card in suit_str[2:] if card in '98765432x']
        ),
        Rule(r"^[AKQJT23456789x]{1}$",  # Matches any single card from the deck (A, K, Q, J, T, 2-9)
            "any",  # Any card is valid
            "Leading from a single card", 
            length_required=1,  # The suit has only 1 card
            max_length=1,  # Maximum length of the suit is 1
            lambda_fn=lambda suit: suit[0]  # Any card in the suit is valid (return the suit as it is)
        ),
        Rule(
            pattern=r"^KJ([2-9x]){1,}$", 
            correct_lead="x",           # Expect a small card lead (4th best is standard)
            explanation="Leading small (4th best) from KJxxx (no Ten) vs NT",
            length_required=5,          # Explicitly require length 5
            max_length=5,               # Explicitly require length 5
            # Lambda confirms 'x' is the expected type.
            # Your main validation logic handles matching 'x' against the actual lead digit ('4').
            lambda_fn=lambda suit_str: ['x']
        ),
        Rule(
            pattern=r"^AJ([2-9x]){1,}$",  # A, J, then exactly THREE small cards (2-9 or 'x')
                                        # This pattern ensures T is not the 3rd card.
            correct_lead="x",            # Expect a small card lead (4th best standard in NT)
            explanation="Leading small (4th best) from AJxxx (no Ten) vs NT",
            length_required=4,           # Explicitly require length 5
            # Lambda confirms 'x' is the expected type.
            # Your main validation logic handles matching 'x' against the actual lead digit.
            lambda_fn=lambda suit_str: [card for card in suit_str[2:] if card in '98765432x']
        )

        ]
    
    # Pick correct rule set
    rules = suit_rules if contract_type == "suit" else nt_rules

    # Ensure SMALL_CARD_RANKS is defined, e.g., at the top of your function or globally
    SMALL_CARD_RANKS = "23456789"

    # Find matching rule
    applicable_rules_explanations = [] # Renamed for clarity, as these are potential explanations if no direct match.
    matched_a_rule = False

    for rule in rules:
        if re.match(rule.pattern, suit):
            matched_a_rule = True
            # Check if length is required
            if rule.length_required and len(suit) < rule.length_required:
                #if verbose: print(f"Rule '{rule.explanation}' skipped: suit length {len(suit)} < required {rule.length_required}")
                continue
            if rule.max_length and len(suit) > rule.max_length:
                #if verbose: print(f"Rule '{rule.explanation}' skipped: suit length {len(suit)} > max {rule.max_length}")
                continue

            expected_options_for_this_rule: list[str] = []
            is_lead_valid_for_this_rule = False

            if rule.lambda_fn:
                raw_lambda_output = rule.lambda_fn(suit)
                # Ensure lambda output is a list of strings
                if isinstance(raw_lambda_output, str):
                    expected_options_for_this_rule = [raw_lambda_output]
                elif isinstance(raw_lambda_output, list):
                    expected_options_for_this_rule = raw_lambda_output
                else:
                    if verbose: print(f"Warning: Lambda for rule '{rule.explanation}' (suit: {suit}) returned invalid type: {type(raw_lambda_output)}. Expected str or list[str].")
                    continue # Skip this rule iteration if lambda output is malformed

                # Now check if the actual lead matches any of the expected options
                for expected_token in expected_options_for_this_rule:
                    if expected_token == lead: # Direct match (e.g., rule expects "2", lead is "2")
                        is_lead_valid_for_this_rule = True
                        break
                    # KEY CHANGE: If rule expects 'x' (a generic small card), 
                    # and actual lead is a specific small card (2-9)
                    if expected_token == 'x' and lead in SMALL_CARD_RANKS:
                        is_lead_valid_for_this_rule = True
                        break
                    # You could add more symbolic interpretations here if needed, e.g.
                    # if expected_token == 'h' and lead in "AKQJT": # 'h' for any honor
                    #    is_lead_valid_for_this_rule = True
                    #    break
            
            else: # No lambda_fn, use rule.correct_lead (which is a string)
                expected_token = rule.correct_lead
                expected_options_for_this_rule = [expected_token] # For consistent message formatting
                
                if expected_token == lead: # Direct match
                    is_lead_valid_for_this_rule = True
                # KEY CHANGE: Also check for 'x' vs actual small card here
                elif expected_token == 'x' and lead in SMALL_CARD_RANKS:
                    is_lead_valid_for_this_rule = True
            
            # After checking all options for this rule:
            if is_lead_valid_for_this_rule:
                # The lead is valid according to this rule.
                # If you have the actual_rank_validator concept, you'd apply it here.
                # For now, let's assume it's valid if the above checks pass.
                # print(f"Lead {lead} for suit {suit} is VALID by rule: {rule.explanation}. Rule expected one of {expected_options_for_this_rule}.")
                return True, [f"Valid lead: {rule.explanation}. (Led {lead}, rule suggests {expected_options_for_this_rule})."]
            else:
                # This rule's pattern matched, but the lead card wasn't one of the expected ones.
                # Store this as a potential explanation if no other rule validates the lead.
                msg = f"Rule: {rule.explanation}. Expected one of: {expected_options_for_this_rule}, but got {lead}."
                #print(f"Lead {lead} for suit {suit} MISMATCHED by rule: {msg}")
                applicable_rules_explanations.append(msg)

    # After checking all rules:
    if not matched_a_rule:
        # No rule perfectly validated the lead, but some rules' patterns matched.
        # These explanations indicate why the lead was wrong according to those rules.
        print(f"{Fore.RED}No opening lead rule found that match suit holding '{suit}' vs {contract_type}. Lead {lead}{Fore.RESET}")
        with open("leads.txt", "a") as file:
            file.write(f"No opening lead rule found that match suit holding '{suit}' vs {contract_type}. Lead {lead}" + "\n")
        return False, applicable_rules_explanations

    # No rule pattern even matched the suit structure.
    # The original code returned True here, which seems like a fallback.
    # It might be better to return False if no rule specifically validates.
    # Let's stick to the original's apparent fallback for now, but flag it.
    #if verbose: print(f"No specific rule pattern matched or validated for suit {suit} with lead {lead} vs {contract_type}. Defaulting to allow.")
    # return True, [f"No matching rule found for suit {suit} vs {contract_type} that explicitly forbids lead {lead}. (Fallback behavior)"]
    # A more standard approach would be:
    return False, [f"No opening lead rule found that validates lead {lead} for suit holding '{suit}' vs {contract_type}."]

