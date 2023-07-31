
class Deal {

    constructor(data) {
        this.data = data

        this.stack = []

        this.position = -1

        this.playIndex = -1
        this.trickIndex = -1
    }

    declarer() {
        if (this.data['contract'] == null) {
            return "Pass"
        }
        return 'NESW'.indexOf(this.data['contract'][this.data['contract'].length - 1])
    }

    next() {
        if (this.position < this.stack.length - 1) {
            this.position += 1
        } else {
            this.pushNext()
        }

        this.renderPosition()
    }

    pushNext() {

        this.playIndex += 1
        
        if (this.playIndex == this.data['play'].length) {
            let trickWinner = this.data['trick_winners'][12]
            this.stack.push(new DealSnapshot(
                this.top().hands,
                this.top().bidding,
                new Trick(trickWinner, []),
                this.top().info,
                new TricksTaken(this.top().tricksTaken.ns + (trickWinner+1) % 2, this.top().tricksTaken.ew + trickWinner % 2)             ))
            return
        }

        if (this.playIndex > this.data['play'].length) {
            return
        }

        let playData = this.data['play'][this.playIndex]

        var player = (this.top().trick.onLead + this.top().trick.cards.length) % 4

        if (this.top().trick.cards.length == 4) {
            this.trickIndex += 1
            let trickWinner = this.data['trick_winners'][this.trickIndex]
            player = (this.declarer() + 1 + trickWinner) % 4
        }

        this.stack.push(this.top().play(player, playData))

        this.position += 1
    }

    prev() {
        if (this.position > 0) {
            this.position -= 1
            this.renderPosition()
        }
    }

    start() {

        let hands = this.data['hands'].split(" ")

        this.stack.push(new DealSnapshot([
                new Hand(hands[0]), // north
                new Hand(hands[1]),
                new Hand(hands[2]),
                new Hand(hands[3])
            ], null, new Trick((this.declarer() + 1) % 4, []), new PlayInfo({}), new TricksTaken(0, 0))
        )
        this.position = 0

        this.renderPosition()
    }

    top() {
        return this.stack[this.position]
    }

    renderPosition() {
        let seats = ["north", "east", "south", "west"]

        for (var i = 0; i < 4; i++) {
            this.top().hands[i].render(document.getElementById(seats[i]))
            this.top().trick.render(document.getElementById("current-trick"))
            this.top().info.render(document.getElementById("info"))
            this.top().tricksTaken.render(document.getElementById("tricks-ns-ew"))
        }
    }
}

class DealSnapshot {

    constructor(hands, bidding, trick, info, tricksTaken) {
        this.hands = hands
        this.bidding = bidding
        this.trick = trick
        this.info = info
        this.tricksTaken = tricksTaken
    }

    play(player, playData) {
        var hands = []
        let card = playData['card']

        for (var i = 0; i < this.hands.length; i++) {
            if (i == player) {
                hands.push(this.hands[i].playCard(card))
            } else {
                hands.push(this.hands[i])
            }
        }

        var trick = this.trick.addCard(card)
        var tt = new TricksTaken(this.tricksTaken.ns, this.tricksTaken.ew)
        if (trick.cards.length > 4) {
            trick = new Trick(player, [card])
            tt.ns += (player + 1) % 2
            tt.ew += player % 2
        }

        return new DealSnapshot(hands, this.bidding, trick, new PlayInfo(playData), tt)
    }

    bid(player, bid) {
        // TODO
    }
}


class Hand {

    constructor(handPBN) {
        let suits = handPBN.split(".")
        this.spades = suits[0]
        this.hearts = suits[1]
        this.diamonds = suits[2]
        this.clubs = suits[3]
    }

    render(element) {
        element.innerHTML = ""

        var html = this.suitHtml("&spades;", this.spades, false)
        html += this.suitHtml("&hearts;", this.hearts, true)
        html += this.suitHtml("&diams;", this.diamonds, true)
        html += this.suitHtml("&clubs;", this.clubs, false)
        
        element.innerHTML = html
    }

    suitHtml(symbol, cards, red) {
        var html = '<div class="suit">'
        if (red) {
            html += '<span class="font-red">'
        } else {
            html += '<span>'
        }
        html += symbol
        html += '</span>\n' // end of symbol

        for (var i = 0; i < cards.length; i++) {
            html = html + '<span class="card">' + cards[i] + '</span>\n'
        }

        html += '<div>'
        return html
    }

    playCard(card) {
        var spades = this.spades
        if (card[0] == "S") {
            spades = spades.replace(card[1], "")
        }
        var hearts = this.hearts
        if (card[0] == "H") {
            hearts = hearts.replace(card[1], "")
        }
        var diamonds = this.diamonds
        if (card[0] == "D") {
            diamonds = diamonds.replace(card[1], "")
        }
        var clubs = this.clubs
        if (card[0] == "C") {
            clubs = clubs.replace(card[1], "")
        }

        return new Hand(spades + '.' + hearts + '.' + diamonds + '.' + clubs)
    }
}


class Trick {

    constructor(onLead, cards) {
        this.onLead = onLead
        this.cards = cards

        this.cardIds = [
            "trick-card-north",
            "trick-card-east",
            "trick-card-south",
            "trick-card-west"
        ]
    }

    addCard(card) {
        let cards = this.cards.slice()
        cards.push(card)

        return new Trick(this.onLead, cards)
    }

    render(element) {
        element.innerHTML = ""

        var html = ""

        let symbols = {
            'S': ['&spades;', false],
            'H': ['&hearts;', true],
            'D': ['&diams;', true],
            'C': ['&clubs;', false]
        }

        for (var i = 0; i < this.cards.length; i++) {
            let cardId = this.cardIds[(this.onLead + i) % 4]
            var cssClass = "trick-card"
            if (i == this.cards.length - 1) {
                cssClass += " highlight"
            }
            html += '<div id="' + cardId + '" class="' + cssClass + '">'

            let cardSymbol = symbols[this.cards[i][0]][0]
            let isRed = symbols[this.cards[i][0]][1]
            let cardVal = this.cards[i][1]

            if (isRed) {
                html += '<span class="font-red">'
            } else {
                html += '<span>'
            }
            html += cardSymbol
            html += '</span>'
            html += '<span>' + cardVal + '</span>'
            html += '</div>'
        }

        element.innerHTML = html
    }
}


class DealerVuln {

    constructor(dealer, vulnNS, vulnEW) {
        this.dealer = dealer
        this.vulnNS = vulnNS
        this.vulnEW = vulnEW
    }

    render(element) {
        element.innerHTML = ""

        var html = ""

        let ids = ["vul-north", "vul-east", "vul-south", "vul-west"]

        for (var i = 0; i < 4; i++) {
            let vId = ids[i]
            var color = "white"
            if (i % 2 == 0 && this.vulnNS || i % 2 == 1 && this.vulnEW) {
                color = "red"
            }

            html += '<div id="' + vId + '" class="' + color + '">'
            if (i == this.dealer) {
                html += '<span class="dealer">D</span>'
            }
            html += '</div>\n'
        }

        element.innerHTML = html

    }
}


class PlayInfo {

    constructor(data) {
        this.data = data
    }

    render(element) {
        element.innerHTML = ""

        var html = ""

        if ("candidates" in this.data) {
            html += '<h3>Candidates</h3>'
            html += '<table>'

            for (var i = 0; i < this.data.candidates.length; i++) {
                let candidate = this.data.candidates[i]

                html += '<tr>'
                html += '<td class="candidate-card">' + candidate['card'] + '</td>'
                if ("expected_tricks" in candidate) {
                    html += '<td>e(tricks)=' + Math.round(candidate['expected_tricks'] * 1000) / 1000 + '</td>'
                }
                if ("expected_score" in candidate) {
                    html += '<td>e(score)=' + Math.round(candidate['expected_score'] * 1000) / 1000 + '</td>'
                }
                if ("insta_score" in candidate) {
                    html += '<td>iscore=' + Math.round(candidate['insta_score'] * 1000) / 1000 + '</td>'
                }
                if ("p_make_contract" in candidate) {
                    html += '<td>e(make)=' + Math.round(candidate['p_make_contract'] * 1000) / 1000 + '</td>'
                }
                html += '</tr>'
            }

            html += '</table>'
        }

        if ("samples" in this.data) {
            html += '<h3>Samples (' + this.data['samples'].length + ')</h3>'
            for (var i = 0; i < this.data['samples'].length; i++) {
                html += '<div>' + this.data['samples'][i] + '</div>'
            }
        }

        element.innerHTML = html
    }
}


class Auction {

    constructor(dealer, bids) {
        this.dealer = dealer
        this.bids = bids

        let nPad = [1, 2, 3, 0]
        this.paddedBids = []
        for (var i = 0; i < nPad[dealer]; i++) {
            this.paddedBids.push("")
        }

        for (var i = 0; i < bids.length; i++) {
            this.paddedBids.push(bids[i]['bid'])
        }
    }

    render(element) {
        element.innerHTML = ""

        var html = ""
        html += '<table>'
        html += '<thead><th>West</th><th>North</th><th>East</th><th>South</th></thead>'
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
            'H': '<span class="font-red">&hearts;</span>',
            'D': '<span class="font-red">&diams;</span>',
            'C': '<span>&clubs;</span>'
        }

        return level + symbolFormat[symbol]
    }

}


class TricksTaken {

    constructor(ns, ew) {
        this.ns = ns
        this.ew = ew
    }

    render(element) {
        element.innerHTML = ""

        var html = ''

        html += '<div id="tricks-ns" class="trick-count"><span>' + this.ns + '</span></div>'
        html += '<div id="tricks-ew" class="trick-count"><span>' + this.ew + '</span></div>'

        element.innerHTML = html
    }
}
