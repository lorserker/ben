function toggleSamples(containerId) {
    const sampleLines = document.getElementById(containerId);
    sampleLines.classList.toggle('hidden');
}

class Deal {

    constructor(data) {
        this.data = data

        this.stack = []

        this.position = -1

        this.playIndex = -1
        this.trickIndex = -1

    }

    board_number() {
        if (this.data['board_number'] == null) {
            return ""
        }
        return this.data['board_number']
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

        // Last card
        if (this.playIndex == this.data['play'].length) {
            let trickstaken
            let trickWinner
            if (typeof this.data['claimed'] !== 'undefined') {
                alert("Claimed " + this.data['claimed'] + " tricks.")
                trickWinner = this.data['trick_winners'][this.data['trick_winners'].length - 1]
                // Trickwinner is relative to declarer
                // declarer is based on position
                if (this.declarer() % 2 == 0) {
                    if (this.data['claimedbydeclarer']) {
                        trickstaken = new TricksTaken(this.top().tricksTaken.ns + this.data['claimed'] + 1, 13 - (this.top().tricksTaken.ns + 1 + this.data['claimed']))
                    } else {
                        trickstaken = new TricksTaken(13 - (this.top().tricksTaken.ew + 1 + this.data['claimed']), this.top().tricksTaken.ew + 1 + this.data['claimed'])
                    }
                } else {
                    if (this.data['claimedbydeclarer']) {
                        trickstaken = new TricksTaken(13 - this.top().tricksTaken.ew + this.data['claimed'], (this.top().tricksTaken.ew + this.data['claimed']))
                    } else {
                        trickstaken = new TricksTaken(this.top().tricksTaken.ew + 1 + this.data['claimed'], 13 - (this.top().tricksTaken.ew + 1 + this.data['claimed']))
                    }
                }
            } else {
                trickWinner = this.data['trick_winners'][12]
                trickstaken = new TricksTaken(this.top().tricksTaken.ns + (trickWinner + 1) % 2, this.top().tricksTaken.ew + (trickWinner) % 2)
            }
            this.stack.push(new DealSnapshot(
                this.top().hands,
                this.top().bidding,
                new Trick(trickWinner, []),
                this.top().info,
                trickstaken))
            this.position += 1
            return
        }

        if (this.playIndex > this.data['play'].length) {
            return
        }

        let playData = this.data['play'][this.playIndex]

        let player = (this.top().trick.onLead + this.top().trick.cards.length) % 4

        if (this.top().trick.cards.length == 4) {
            this.trickIndex += 1
            let trickWinner = this.data['trick_winners'][this.trickIndex]
            player = (this.declarer() + 1 + trickWinner) % 4
        }

        this.stack.push(this.top().play(player, playData, this.declarer()))

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

        for (let i = 0; i < 4; i++) {
            this.top().hands[i].render(document.getElementById(seats[i]))
            this.top().trick.render(document.getElementById("current-trick"))
            this.top().info.render(document.getElementById("info"), i)
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

    play(player, playData, declarer) {
        let hands = []
        let card = playData['card']

        for (let i = 0; i < this.hands.length; i++) {
            if (i == player) {
                hands.push(this.hands[i].playCard(card))
            } else {
                hands.push(this.hands[i])
            }
        }

        let trick = this.trick.addCard(card)
        let tt = new TricksTaken(this.tricksTaken.ns, this.tricksTaken.ew)
        if (trick.cards.length > 4) {
            trick = new Trick(player, [card])
            tt.ns += (player + 1) % 2
            tt.ew += player % 2
        }

        return new DealSnapshot(hands, this.bidding, trick, new PlayInfo(playData, player, declarer), tt)
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

        let html = this.suitHtml("&spades;", this.spades, false)
        html += this.suitHtml("&hearts;", this.hearts, true)
        html += this.suitHtml("&diams;", this.diamonds, true)
        html += this.suitHtml("&clubs;", this.clubs, false)

        element.innerHTML = html
    }

    suitHtml(symbol, cards, red) {
        let html = '<div class="suit">'
        if (red) {
            html += '<span class="font-red">'
        } else {
            html += '<span>'
        }
        html += symbol
        html += '</span>\n' // end of symbol

        for (const element of cards) {
            html = html + '<span class="card">' + element + '</span>\n'
        }

        html += '<div>'
        return html
    }

    playCard(card) {
        let spades = this.spades
        if (card[0] == "S") {
            spades = spades.replace(card[1], "")
        }
        let hearts = this.hearts
        if (card[0] == "H") {
            hearts = hearts.replace(card[1], "")
        }
        let diamonds = this.diamonds
        if (card[0] == "D") {
            diamonds = diamonds.replace(card[1], "")
        }
        let clubs = this.clubs
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

        let html = ""

        let symbols = {
            'S': ['&spades;', false],
            'H': ['&hearts;', true],
            'D': ['&diams;', true],
            'C': ['&clubs;', false]
        }

        for (let i = 0; i < this.cards.length; i++) {
            let cardId = this.cardIds[(this.onLead + i) % 4]
            let cssClass = "trick-card"
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

        let html = ""

        let ids = ["vul-north", "vul-east", "vul-south", "vul-west"]

        for (let i = 0; i < 4; i++) {
            let vId = ids[i]
            let color = "white"
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

    constructor(data, player, declarer) {
        this.data = data
        this.player = player
        this.declarer = declarer
    }

    render(element, index) {
        if (this.player != index)
            return
        element.innerHTML = ""

        let html = ""

        if ("candidates" in this.data && this.data.candidates.length > 0) {
            html += '<h3>Candidates</h3>'
            html += '<table>'

            for (const element of this.data.candidates) {
                let candidate = element

                html += '<tr>'
                html += '<td class="candidate-card">' + candidate['card'] + '</td>'
                if ("expected_tricks" in candidate) {
                    html += '<td>e(tricks)=' + Math.round(candidate['expected_tricks'] * 100) / 100 + '</td>'
                }
                if ("expected_tricks_sd" in candidate) {
                    html += '<td>e(tricks)(SD)=' + Math.round(candidate['expected_tricks_sd'] * 100) / 100 + '</td>'
                }
                if ("expected_tricks_dd" in candidate) {
                    html += '<td>e(tricks)(DD)=' + Math.round(candidate['expected_tricks_dd'] * 100) / 100 + '</td>'
                }
                if ("expected_score" in candidate) {
                    html += '<td>e(score)=' + Math.round(candidate['expected_score'] * 100) / 100 + '</td>'
                }
                if ("expected_score_sd" in candidate) {
                    html += '<td>e(score)sd=' + Math.round(candidate['expected_score_sd'] * 100) / 100 + '</td>'
                }
                if ("expected_score_dd" in candidate) {
                    html += '<td>e(score)dd=' + Math.round(candidate['expected_score_dd'] * 100) / 100 + '</td>'
                }
                if ("insta_score" in candidate) {
                    html += '<td>iscore=' + Math.round(candidate['insta_score'] * 100) / 100 + '</td>'
                }
                if ("p_make_contract" in candidate) {
                    html += '<td>e(make/set)=' + Math.round(candidate['p_make_contract'] * 100) / 100 + '</td>'
                }
                html += '</tr>'
            }

            html += '</table>'
        }

        if ("hcp" in this.data && "shape" in this.data) {
            if (this.data['hcp'] != -1 && this.data['shape'] != -1) {
                let shape = this.data['shape'].reduce((acc, val) => acc.concat(val), []);
                html += '<h3>Bidding Info</h3>'
                if (this.data['hcp'].length > 2) {
                    html += '<div>Dummy: ' + this.data['hcp'][0] + ' hcp, shape: '
                    for (let i = 0; i < 4; i++) {
                        html += shape[i] + " "
                    }
                    html += '</div>'
                    html += '<div>Partner: ' + this.data['hcp'][1] + ' hcp, shape: '
                    for (let i = 0; i < 4; i++) {
                        html += shape[i + 4] + " "
                    }
                    html += '</div>'
                    html += '<div>Declarer: ' + this.data['hcp'][2] + ' hcp, shape: '
                    for (let i = 0; i < 4; i++) {
                        html += shape[i + 8] + " "
                    }
                    html += '</div>'
                } else {
                    // we are seated after declarer
                    if ((this.player - this.declarer + 4) % 4 == 1) {
                        html += '<div>Declarer: ' + this.data['hcp'][0] + ' hcp, shape: '
                        for (let i = 0; i < 4; i++) {
                            html += shape[i] + " "
                        }
                        html += '</div>'
                        html += '<div>Partner: ' + this.data['hcp'][1] + ' hcp, shape: '
                        for (let i = 0; i < 4; i++) {
                            html += shape[i + 4] + " "
                        }
                        html += '</div>'
                    } else
                        // we are seated before declarer
                        if ((this.player - this.declarer + 4) % 4 == 3) {
                            html += '<div>Partner: ' + this.data['hcp'][0] + ' hcp, shape: '
                            for (let i = 0; i < 4; i++) {
                                html += shape[i] + " "
                            }
                            html += '</div>'
                            html += '<div>Declarer: ' + this.data['hcp'][1] + ' hcp, shape: '
                            for (let i = 0; i < 4; i++) {
                                html += shape[i + 4] + " "
                            }
                            html += '</div>'
                        }
                        else {
                            // RHO
                            if (this.player == 0) {
                                html += '<div>West: ' + this.data['hcp'][0] + ' hcp, shape: '
                            }
                            if (this.player == 1) {
                                html += '<div>North: ' + this.data['hcp'][0] + ' hcp, shape: '
                            }
                            if (this.player == 2) {
                                html += '<div>East: ' + this.data['hcp'][0] + ' hcp, shape: '
                            }
                            if (this.player == 3) {
                                html += '<div>South: ' + this.data['hcp'][0] + ' hcp, shape: '
                            }

                            for (let i = 0; i < 4; i++) {
                                html += shape[i] + " "
                            }
                            html += '</div>'
                            // LHO
                            if (this.player == 0) {
                                html += '<div>East: ' + this.data['hcp'][1] + ' hcp, shape: '
                            }
                            if (this.player == 1) {
                                html += '<div>South: ' + this.data['hcp'][1] + ' hcp, shape: '
                            }
                            if (this.player == 2) {
                                html += '<div>West: ' + this.data['hcp'][1] + ' hcp, shape: '
                            }
                            if (this.player == 3) {
                                html += '<div>North: ' + this.data['hcp'][1] + ' hcp, shape: '
                            }
                            for (let i = 0; i < 4; i++) {
                                html += shape[i + 4] + " "
                            }
                            html += '</div>'
                        }
                }
            }
        }

        if ("quality" in this.data) {
            html += '<br><strong>Sample quality:</strong> ' + this.data['quality']
        }

        if ("samples" in this.data && this.data['samples'].length > 0) {
            html += `
                <h3 class="samples" onclick="toggleSamples('sampleLinesPlay')"><strong>Samples(${this.data['samples'].length}):</strong></h3>
                <div id="sampleLinesPlay" class="hidden">
                <ul>${this.data.samples.map(sample => `<li>${sample.replace(/\n/g, "<br>")}</li>`).join('')}</ul>
                </div>
                `
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
        for (let i = 0; i < nPad[dealer]; i++) {
            this.paddedBids.push("")
        }

        for (const element of bids) {
            this.paddedBids.push(element['bid'])
        }
        this.auctionString = ''
        for (const element of bids) {
            this.auctionString += element['bid'].replace("PASS", "P") + " "
        }
    }

    render(element) {
        element.innerHTML = ""

        let html = ""
        html += '<table>'
        html += '<thead><th>West</th><th>North</th><th>East</th><th>South</th></thead>'
        html += '<tbody>'

        for (let i = 0; i < this.paddedBids.length; i++) {
            if (i % 4 == 0) {
                html += '<tr>'
            }

            html += '<td>' + this.formatBid(this.paddedBids[i]) + '</td>'

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

        let html = ''

        html += '<div id="tricks-ns" class="trick-count"><span>' + this.ns + '</span></div>'
        html += '<div id="tricks-ew" class="trick-count"><span>' + this.ew + '</span></div>'

        element.innerHTML = html
    }
}
