
/* Messages

- card? (player)
- card played (player, card)
- confirm?
- show all (all hands)

*/


/* Deal - state

dealer: {0, 1, 2, 3, 4}
vuln: [v_ns, v_ew]  
auction
contract: Contract
my_hand: Hand
public_hand: Hand (unknown during bidding and lead)
played_tricks: List[Trick]
current_trick: Trick
tricks_ns: Int
tricks_ew: Int

whose turn?

*/


/* Controller

- deal
- socket

- handle received messages


*/


/* UI handler


- handle clicks (user input)
    - update controller about events

- render stuff on screen
    - controller will tell us what happened
    - we know what to update
        
*/