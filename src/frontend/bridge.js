

class Card {

    constructor(symbol) {
        this.symbol = symbol
        this.suit = 'SHDC'.indexOf(symbol[0])
        this.rank = symbol[1]
        this.value = 'AKQJT98765432'.indexOf(symbol[1])
    }

    render(element) {
        let card = document.createElement('div')
        card.classList.add('card')
        card.setAttribute('data-value', this.rank)
        card.setAttribute('symbol', this.symbol)
        card.innerHTML = ['&spades;', '&hearts;', '&diams;', '&clubs;'][this.suit]
        if (this.suit == 1 || this.suit == 2) {
            card.classList.add('red')
        }

        element.appendChild(card)
    }

}


class Hand {

    constructor(cards) {
        this.cards = cards

        this.suits = [[], [], [], []]

        for (var i = 0; i < cards.length; i++) {
            this.suits[cards[i].suit].push(cards[i])
        }
    }

    isPlayable(card, trick) {
        if (!this.hasCard(card)) {
            return false
        }
        if (trick.cards.length == 0) {
            return true
        }
        if (trick.cards.length >= 4) {
            return false
        }
        let leadSuit = trick.cards[0].suit
        if (this.suits[leadSuit].length == 0) {
            return true
        }
        return card.suit == leadSuit
    }

    hasCard(card) {
        var found = false;
        for (var i = 0; i < this.cards.length; i++) {
            if (this.cards[i].symbol == card.symbol) {
                found = true
                break
            }
        }
        return found
    }

    // returns a new hand with the played card removed
    play(card) {
        var remainingCards = []
        for (var i = 0; i < this.cards.length; i++) {
            if (card.symbol != this.cards[i].symbol) {
                remainingCards.push(this.cards[i])
            }
        }
        return new Hand(remainingCards)
    }

    render(element) {
        element.textContent = ''

        this.suits[0].forEach(c => c.render(element))
        this.suits[1].forEach(c => c.render(element))
        this.suits[3].forEach(c => c.render(element))
        this.suits[2].forEach(c => c.render(element))
    }

    renderEW(element) {
        element.innerHTML = ""

        var html = '<div>'
        html += this.suitHtml("&spades;", this.suits[0].map(c => c.rank), false)
        html += this.suitHtml("&hearts;", this.suits[1].map(c => c.rank), true)
        html += this.suitHtml("&diams;", this.suits[2].map(c => c.rank), true)
        html += this.suitHtml("&clubs;", this.suits[3].map(c => c.rank), false)
        html += '</div>'
        
        element.innerHTML = html
    }

    suitHtml(symbol, cards, red) {
        var html = '<div class="suit">'
        if (red) {
            html += '<span class="red">'
        } else {
            html += '<span>'
        }
        html += symbol
        html += '</span>\n' // end of symbol

        for (var i = 0; i < cards.length; i++) {
            html = html + '<span class="card-ew">' + cards[i] + '</span>\n'
        }

        html += '</div>'
        return html
    }

}


function parseHand(pbnString) {
    let suits = pbnString.split('.')
    let cards = []
    let suitSymbols = 'SHDC'

    for (var i = 0; i < suits.length; i++) {
        for (var j = 0; j < suits[i].length; j++) {
            cards.push(new Card(suitSymbols[i] + suits[i][j]))
        }
    }

    return new Hand(cards)
}


class Contract {
    // TODO: implement
    constructor() {

    }

    level() {

    }

    trumpSuit() {

    }

    isNoTrump() {

    }

    isDoubled() {

    }

    isRedoubled() {

    }
}


class Trick {

    constructor(leadPlayer, cards) {
        this.leadPlayer = leadPlayer
        this.cards = cards
    }

    isComplete() {
        return this.cards.length == 4
    }

    winner(strain) {
        if (this.isComplete()) {
            let trump = strain - 1

            var trumpPlayed = false
            if (trump >= 0) {
                for (var i = 0; i < this.cards.length; i++) {
                    if (this.cards[i].suit == trump) {
                        trumpPlayed = true
                        break
                    }
                }
            }

            var bestValue = 100
            var bestIndex = -1

            if (trumpPlayed) {
                for (var i = 0; i < this.cards.length; i++) {
                    if (this.cards[i].suit != trump) {
                        continue
                    }
                    if (this.cards[i].value < bestValue) {
                        bestValue = this.cards[i].value
                        bestIndex = i
                    }
                }
            } else {
                let ledSuit = this.cards[0].suit
                for (var i = 0; i < this.cards.length; i++) {
                    if (this.cards[i].suit != ledSuit) {
                        continue
                    }
                    if (this.cards[i].value < bestValue) {
                        bestValue = this.cards[i].value
                        bestIndex = i
                    }
                }
            }

            return (this.leadPlayer + bestIndex) % 4
        }
    }

    render(slotElements) {
        slotElements.forEach(el => el.textContent = '')

        for (var i = this.leadPlayer, j = 0; j < this.cards.length; i = (i + 1) % 4, j++) {
            this.cards[j].render(slotElements[i])
        }
    }

}


class Deal {

    constructor(dealer, vuln, hand) {
        this.dealer = dealer
        this.vuln = vuln
        this.hand = hand
        this.tricksCount = [0, 0]

        this.public = undefined

        this.turn = this.dealer
        this.auction = []
        this.tricks = []
        this.currentTrick = undefined

        this.expectCardInput = false
        this.expectTrickConfirm = false
        this.expectBidInput = false

        this.canDouble = false
        this.canRedouble = false
    }

    renderTricks(element) {
        element.textContent = ''
        element.innerHTML = 'Tricks NS:' + this.tricksCount[0] + ' EW:' + this.tricksCount[1]
    }

    renderAuction(element) {
        element.textContent = ''

        var html = ''
        html += '<div>'
        html += '<div id="auction-container"></div>'
        html += '<div class="tricks"></div>'
        html += '</div>'

        element.innerHTML = html

        this.renderTricks(document.querySelector('.tricks'))

        let auction = new Auction(this.dealer, this.vuln, this.auction)

        auction.render(document.querySelector('#auction-container'))
    }

    renderBiddingBox(element) {
        element.textContent = ''

        let auct = new Auction(this.dealer, this.vuln, this.auction)

        var html = ''
        html += '<div id="bidding-box">'
        html += '<div id="bidding-levels">'
        let minBiddableLevel = auct.getMinimumBiddableLevel()
        for (var i = 1; i <= 7; i++) {
            if (i < minBiddableLevel) {
                html += '<div class="invalid">' + i + '</div>'
            } else {
                html += '<div>' + i + '</div>'
            }
        }
        html += '</div>'

        html += '<div id="bidding-suits" class="hidden">'
        html += '<div class="bid-clubs" symbol="C">&clubs;</div>'
        html += '<div class="bid-diamonds red" symbol="D">&diams;</div>'
        html += '<div class="bid-hearts red" symbol="H">&hearts;</div>'
        html += '<div class="bid-spades" symbol="S">&spades;</div>'
        html += '<div class="bid-nt" symbol="N">NT</div>'
        html += '</div>'

        html += '<div id="bidding-calls">'
        html += '<div class="pass">PASS</div>'
        if (this.canDouble) {
            html += '<div class="double">X</div>'
        } else {
            html += '<div class="double invalid">X</div>'
        }
        if (this.canRedouble) {
            html += '<div class="redouble">XX</div>'
        } else {
            html += '<div class="redouble invalid">XX</div>'
        }
        html += '</div>'

        html += '</div>'

        element.innerHTML = html
    }

    selectBiddingLevel(level) {

    }
}


class Auction {

    constructor(dealer, vuln, bids) {
        this.dealer = dealer
        this.vuln = vuln
        this.bids = []

        for (var i = 0; i < bids.length; i++) {
            if (bids[i] != 'PAD_START') {
                this.bids.push(bids[i])
            }
        }

        let nPad = [1, 2, 3, 0]
        this.paddedBids = []
        for (var i = 0; i < nPad[dealer]; i++) {
            this.paddedBids.push("")
        }

        for (var i = 0; i < this.bids.length; i++) {
            this.paddedBids.push(this.bids[i])
        }
    }

    getMinimumBiddableLevel() {
        for (var i = this.bids.length - 1; i >= 0; i--) {
            let level = parseInt(this.bids[i][0])
            if (isNaN(level)) {
                continue
            }
            if (this.bids[i][1] == "N") {
                return level + 1
            }
            return level
        }
        return 1
    }

    getMinBiddableSuitForLevel(level) {
        for (var i = this.bids.length - 1; i >= 0; i--) {
            let lastBidLevel = parseInt(this.bids[i][0])
            if (isNaN(lastBidLevel)) {
                continue
            }
            if (lastBidLevel < level) {
                return 0
            }
            let suitIndex = 'CDHSN'.indexOf(this.bids[i][1])
            return suitIndex + 1
        }

        return 0
    }

    canDouble() {
        return false
    }

    canRedouble() {
        return false
    }

    render(element) {
        element.innerHTML = ""

        var html = '<div id="auction">'
        html += '<table>'
        html += '<thead>'
        if (this.vuln[1]) {
            html += '<th class="red">West</th>'
        } else {
            html += '<th>West</th>'
        }
        if (this.vuln[0]) {
            html += '<th class="red">North</th>'
        } else {
            html += '<th>North</th>'
        }
        if (this.vuln[1]) {
            html += '<th class="red">East</th>'
        } else {
            html += '<th>East</th>'
        }
        if (this.vuln[0]) {
            html += '<th class="red">South</th>'
        } else {
            html += '<th>South</th>'
        }
        html += '</thead>'
        html += '<tbody>'

        for (var i = 0; i < this.paddedBids.length; i++) {
            if (i % 4 == 0) {
                html += '<tr>'
            }

            html +='<td>' + this.formatBid(this.paddedBids[i]) + '</td>'

            if (i % 4 == 3) {
                html += '</tr>\n'
            }
        }

        html += '</tbody>'
        html += '</table>'
        html += '</div>'

        element.innerHTML = html
    }

    formatBid(bid) {
        let calls = {
            '': '',
            'PASS': 'p',
            'X': 'X',
            'XX': 'XX'
        }

        if (bid in calls) {
            return calls[bid]
        }

        let level = bid[0]
        let symbol = bid[1]

        let symbolFormat = {
            'N': 'NT',
            'S': '<span>&spades;</span>',
            'H': '<span class="red">&hearts;</span>',
            'D': '<span class="red">&diams;</span>',
            'C': '<span>&clubs;</span>'
        }

        return level + symbolFormat[symbol]
    }

}
