<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEN - The oracle</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <link rel="stylesheet" href="/app/style.css">
    <link rel="stylesheet" href="/app/viz.css">
    <script src="/app/samplesTable.js"></script>

    <!-- Include jQuery and jQuery UI -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.min.js"></script>
</head>

<body>
    <div>
        <h1>API for BEN. Version 0.8.6.9</h1>
    </div>
    <div id="loader"></div> 
    <div id="dealdiv">
        <h1>Enter Board information</h1>
        <label for="userInput">Your name:</label>
        <input type="text" id="userInput" placeholder="Enter user" required oninput="saveUser()"><br><br>

        <label for="seatInput">Select your seat:</label>
        <select id="seatInput">
            <option value="N">North</option>
            <option value="S" selected>South</option>
            <option value="E">East</option>
            <option value="W">West</option>
        </select><br><br>

        
        <button onclick="readImportedFile()" id="importBtn" disabled>Import file</button>&nbsp;&nbsp;<input type="file" accept=".txt, .pbn, .lin" id="importFile" onchange="enableImportBtn()"><br><br>

        
        <textarea id="importInput" rows="10" cols="64"></textarea>
        <button onclick="processPbnInput()">Parse data as PBN</button>&nbsp;&nbsp;&nbsp;
        <button onclick="processLinFile()">Parse data as LIN</button><br><br>

        <div id="parsedData" style="display: hidden"></div>

        <input type="checkbox" id="old_server" data-default="false"><label for="old_server">Old model</label><br>
        <input type="checkbox" id="details" data-default="true" checked><label for="details">Details</label><br>
        <input type="checkbox" id="matchpoint" data-default="false"><label for="matchpoint">Matchpoint</label><br>
        <input type="checkbox" id="explain" data-default="false"><label for="explain">Explain bid</label><br>
    
        <label for="dealerInput">Dealer:</label>
        <select id="dealerInput">
            <option value="N" selected>North</option>
            <option value="S">South</option>
            <option value="E">East</option>
            <option value="W">West</option>
        </select><br>

        <label for="vulInput">Vulnerable:</label>
        <select id="vulInput">
            <option value="">None</option>
            <option value="@v">NS</option>
            <option value="@V">EW</option>
            <option value="@v@V">All</option>
        </select><br>

        <label for="biddingInput">Bidding:</label>
        <input type="text" id="biddingInput" size="64" placeholder="Enter bidding (1S-P-2H-P-P-P)" onmouseover="showTooltip(3)" onmouseout="hideTooltip(3)"><br>
        <small id="tooltip3" style="display: none;">Enter bids separated by hyphens. Allowed bids: 1C, 1D, 1H, 1S, 1N, ..., 7N, X, XX, P</small><br><br>

        <label for="handInput">Hand:</label>
        <input type="text" id="handInput" size="40"  placeholder="Enter hand (ie AKJT.QT9.J987.5432)" required onmouseover="showTooltip(1)" onmouseout="hideTooltip(1)"><br>

        <label for="dummyInput">Dummy/Declarer:</label>
        <input type="text" id="dummyInput" size="40" placeholder="Enter hand (e.g., AKJT.QT9.J987.5432)" required onmouseover="showTooltip(1)" onmouseout="hideTooltip(1)"><br>
        <small id="tooltip1" style="display: none;">Please enter the hand as four suits separated by '.'</small><br>
            
        <label for="playInput">Play:</label>
        <input type="checkbox" id="format" data-default="false"><label for="format">Play is in PBN format (opening leader first in all tricks, but incomplete tricks should follow lead rules)</label><br>
        <textarea id="playInput"   rows="3" cols="64" placeholder="Enter cards (S7H3H6)" required onmouseover="showTooltip(2)" onmouseout="hideTooltip(2)"></textarea><br>
        <small id="tooltip2" style="display: none;">Please enter all the played cards including cards from BEN</small><br><br>

                    
        <button onclick="callAPI('bid')">Ask BEN for bid</button>&nbsp;&nbsp;
        <button onclick="callAPI('play')">Ask BEN for play</button><br><br>
        <button onclick="callAPI('lead')">Ask BEN for lead</button><br><br>
        <button onclick="callAPI('claim')">Claim</button><br><br>
        <button onclick="callAPI('contract')">Ask BEN for contract</button>&nbsp;&nbsp;
        <button onclick="callAPI('explain')">Explain last bid</button><br><br>
        <button onclick="callAPI('explain_auction')">Explain auction</button><br><br>
        <div id="result"></div><br><br>
        <div id="info"></div><br><br>
        <div><a href="/home">Home</a></div>
        </div>
        <br>


    <script>

        const suitchars = 'CDHSN';
        const cardchars = '23456789TJQKA';
        const seatIndices = {'N': 0, 'E': 1, 'S': 2, 'W': 3};
        let hands = [];
        var howManyCards = new Array(4);
        var howManyCardsDealt = new Array(4);
        var howManySuit = new Array(4);
        var howManySuitDealt = new Array(4);
        var howManyXs = new Array(4);
        var deck = new Array(4);
        var openinglead = ''


        window.addEventListener('DOMContentLoaded', (event) => {
            enableImportBtn();
        });

        function enableImportBtn() {
            const btn = document.getElementById('importBtn');
            const fileInput = document.getElementById('importFile');
            btn.disabled = !(fileInput.files.length > 0);
        }

        function extractValue(s) {
            const openingQuoteIndex = s.indexOf('"');
            const closingQuoteIndex = s.lastIndexOf('"');

            if (openingQuoteIndex !== -1 && closingQuoteIndex !== -1 && closingQuoteIndex > openingQuoteIndex) {
                return s.substring(openingQuoteIndex + 1, closingQuoteIndex);
            } else {
                throw new Error(`Invalid string format: "${s}"`);
            }
        }

        function convertHandLineToArray(handLine) {
            const [seatStr, handsArr] = handLine.split(':');

            const handsNesw = [''];
            const index = seatIndices[seatStr];

            handsArr.split(' ').forEach((hand, idx) => {
                handsNesw[(index + idx) % 4] = hand;
            });

            return handsNesw;
        }
        function processLinFile() {
            bidSequence = [];
            bidSeqPoint = 0;
            openinglead = ''
            document.getElementById('biddingInput').value = ''
            document.getElementById('playInput').value = ''
            // Handle firefox lin-link
            let lin = decodeURI(document.getElementById('importInput').value).replace(/%2C/g,",");

            var startIndex = 0;
            parts = lin.split('=')
            if (parts.length > 1) {
                lin = parts[parts.length-1].trim();
            } 
            while (startIndex < lin.length) {
                var openPipeIndex = lin.indexOf('|', startIndex);
                if (openPipeIndex < 2) break;
                var closePipeIndex = lin.indexOf('|', openPipeIndex + 1);
                if (closePipeIndex < 0) break;

                var command = lin.substring(openPipeIndex - 2, openPipeIndex).toUpperCase();
                var param = lin.substring(openPipeIndex + 1, closePipeIndex, 1);

                processLinCommand(command, param);
                startIndex = closePipeIndex + 1;
            }
            // Replace "D" with "X" and "RD" with "XX"
            bidSequence = bidSequence.map(item => {
                if (item === "D") {
                    return "X";
                } else if (item === "R") {
                    return "XX";
                }
                return item; // Keep other items unchanged
            });
            document.getElementById('biddingInput').value = bidSequence.join('-').toUpperCase()
            const seat = document.getElementById('seatInput').value;
            const dealer = "NESW".indexOf(document.getElementById('dealerInput').value);
            // Deck is based from South
            const player = ("SWNE".indexOf(seat)) % 4
            var openingsuit = suitchars.indexOf(openinglead[0])
            var openingcard = cardchars.indexOf(openinglead[1])
            var lefty = deck[openingsuit][openingcard]
            var dummy = (lefty + 1) % 4
            document.getElementById('handInput').value = ''
            document.getElementById('dummyInput').value = ''
            for (let i = 3; i >= 0; i--) {
                for (let j = 12; j >= 0; j--) {
                    if ((player) === deck[i][j]) {
                        document.getElementById('handInput').value += cardchars[j]
                    }
                    if (dummy === deck[i][j]) {
                        document.getElementById('dummyInput').value += cardchars[j]
                    }
                }
                if (i > 0) {
                    document.getElementById('handInput').value += '.'
                    document.getElementById('dummyInput').value += '.'
                }
            }
            // Get the checkbox element
            const checkbox = document.getElementById('format');
            
            // Set the checkbox to true (checked)
            checkbox.checked = false;

            // Optional: Update the data-default attribute if needed
            checkbox.setAttribute('data-default', 'false');

        }

        function processLinCommand(command, param) {
            switch (command) {
                case 'DT':
                    // Ignore here
                    break;
                case 'SV':
                    value = 'None'
                    param = param.toUpperCase()
                    if (param === 'N') {
                        value = 'NS'
                    }
                    if (param === 'E') {
                        value = 'EW'
                    }
                    if (param === 'B') {
                        value = 'All'
                    }
                    // Select the option node with text equal to "All"

                    let vulOptions = document.querySelectorAll("#vulInput option");
                    for (const element of vulOptions) {
                        if (element.text == value) {
                            element.selected = true;
                            break;
                        }
                    }
                    break;
                case 'MD':
                    deal(param);
                    document.getElementById('dealerInput').value = "SWNE"[parseInt(param.charAt(0)) - 1]
                    break;
                case 'SK':
                    // Ignore here
                    break;
                case 'MB':
                    processBidding(param);
                    break;
                case 'PC':
                    if (openinglead == "") {
                        openinglead = param.toUpperCase()
                    }
                    document.getElementById('playInput').value += param.replace(/\s+/g, '') + " ";
                    break;
                case 'AN':
                    // Ignore here
                    break;
                case 'AH':
                    // Ignore here
                    break;
                case 'PN':
                    // Ignore here
                    break;
                case 'MC':
                    // Ignore here
                    break;
                case 'NT':
                    // Ignore here
                    break;
                case 'AT':
                    // Ignore here
                    break;
                case 'ST':
                    // Ignore here
                    break;
                case 'RH':
                    // Ignore here
                    break;
            }
        }

        function processBidding(bidding) {
            var i = 0;

            while (i < bidding.length) {
                var c = bidding.charAt(i).toUpperCase();
                var len = 0;

                if (c == '-') {
                    i++;
                    continue;
                }
                if (c == 'P' || c == 'D' || c == 'R' || c == '?') {
                    len = 1;
                } else if (c == 'X') {
                    len = 1;

                    if (i < bidding.length - 1 && bidding.charAt(i + 1).toUpperCase() == 'X') {
                        len++;
                    }
                } else if (c >= '1' && c <= '7') {
                    len = 2;
                }
                if (len == 0) {
                    break;
                }
                bidSequence[bidSeqPoint] = bidding.substring(i, i + len);
                bidSeqMax = bidSeqPoint;
                bidSeqPoint++;
                if (i + 1 < bidding.length - 1 && bidding.charAt(i + len) == '!') {
                    len++;
                }
                i += len;
            }
        }

        function deal(dealString) {
            if (!dealString || dealString.length == 0) return false;

            clearDeck();

            var seat = 0;
            var suit = -1;
            var card = -1;
            var p = 1;

            while (p < dealString.length) {
                var ch = dealString.charAt(p).toUpperCase();

                if (ch == ',') {
                    seat++;
                    if (seat > 3) return false;
                    suit = -1;
                    card = -1;
                }

                var st = suitchars.indexOf(ch);

                if (st >= 0) suit = st;

                if (ch == 'X') {
                    if (suit < 0) return false;
                    if (howManyCardsDealt[seat] < 13) {
                        howManyXs[seat][suit]++;
                        howManyCards[seat]++;
                        howManyCardsDealt[seat]++;
                        howManySuit[seat][suit]++;
                        howManySuitDealt[seat][suit]++;
                    }
                } else {
                    if (ch == '1') {
                        card = 8;
                    } else {
                        card = cardchars.indexOf(ch);
                    }

                    if (card >= 0) {
                        if (suit < 0) return false;
                        dealCardToPlayer(suit, card, seat);
                    }
                }
                p++;
            }
            fillInFourthHand();
            return true;
        }

        function fillInFourthHand() {
            var num13 = 0;
            var not13 = -1;

            for (seat = 0; seat < 4; seat++) {
                if (howManyCardsDealt[seat] == 13) num13++;
                else not13 = seat;
            }

            if (num13 == 3 && not13 >= 0) {
                for (suit = 0; suit < 4; suit++) {
                    for (card = 0; card < 13; card++) {
                        if (deck[suit][card] == -10) dealCardToPlayer(suit, card, not13);
                    }
                }
                return true;
            }
            return false;
        }

        function dealCardToPlayer(suit, card, seat) {
            var who = deck[suit][card];

            if (who == -10) {
                deck[suit][card] = seat;
                howManyCards[seat]++;
                howManyCardsDealt[seat]++;
                howManySuit[seat][suit]++;
                howManySuitDealt[seat][suit]++;
            }
        }

        function clearDeck() {
            for (w = 0; w < 4; w++) {
                howManyCards[w] = 0;
                howManyCardsDealt[w] = 0;
                howManySuit[w] = new Array(4);
                howManySuitDealt[w] = new Array(4);
                howManyXs[w] = new Array(4);
                for (s = 0; s < 4; s++) {
                    howManySuit[w][s] = 0;
                    howManySuitDealt[w][s] = 0;
                    howManyXs[w][s] = 0;
                }
            }

            for (s = 0; s < 4; s++) {
                deck[s] = new Array(13);
                for (c = 0; c < 13; c++) {
                    deck[s][c] = -10;
                }
            }
        }

        function processPbnInput() {
            const clipboardContent = document.getElementById('importInput').value;
            const lines = clipboardContent.split(/[\r\n]+/u);
            const parsedData = [];
            let currentBlock = null;

            let tokenKey = ''
            document.getElementById('biddingInput').value = ''
            document.getElementById('playInput').value = ''
            lines.forEach((line) => {
                const tokens = line.split(/\s+/);

                if (tokens[0] === '[Dealer') {
                    currentBlock = { key: 'Dealer', value: extractValue(line) };
                    parsedData.push(currentBlock);
                    document.getElementById('dealerInput').value = currentBlock.value
                } 
                if (tokens[0] === '[Declarer') {
                    var declarer = extractValue(line)
                    currentBlock = { key: 'Declarer', value: declarer };
                    parsedData.push(currentBlock);
                    if (declarer) {
                        const dummy = ("NESW".indexOf(currentBlock.value) + 2) % 4
                        document.getElementById('dummyInput').value = hands[dummy]
                    }

                } 
                if (tokens[0] === '[Auction') {
                    tokenKey = "Auction"
                    currentBlock = { key: 'Auction', value: extractValue(line) };
                    parsedData.push(currentBlock);
                    return;
                }                 
                if (tokens[0] === '[Play') {
                    tokenKey = "Play"
                    currentBlock = { key: 'Play', value: extractValue(line) };
                    parsedData.push(currentBlock);
                    return;
                }                 
                if (tokens[0] === '[Contract') {
                    currentBlock = { key: 'Contract', value: extractValue(line) };
                    parsedData.push(currentBlock);
                }                 
                if (tokens[0] === '[Vulnerable') {
                    currentBlock = { key: 'Vulnerable ', value: extractValue(line) };
                    parsedData.push(currentBlock);
                    document.getElementById("vulInput").value = currentBlock.value;

                    // Select the option node with text equal to "All"
                    let vulOptions = document.querySelectorAll("#vulInput option");
                    for (const option of vulOptions) {
                        // Check if the visible text matches
                        if (option.text === currentBlock.value) {
                            // Set 'selected' property on the matching option
                            option.selected = true;
                            break;  // No need to continue the loop after a match
                        } else {
                            // Ensure other options are not selected
                            option.selected = false;
                        }
                    }
                } 
                
                if (tokens[0] === '[Deal') {
                    currentBlock = { key: 'Deal', value: extractValue(line) };
                    parsedData.push(currentBlock);
                    hands = convertHandLineToArray(currentBlock.value)
                    const seat = document.getElementById('seatInput').value;
                    const player = "NESW".indexOf(seat)
                    document.getElementById('handInput').value = hands[player]
                } 

                if (tokenKey && !tokens[0].startsWith("[")) {
                    if (tokenKey === 'Auction') {
                        line = line.replace(/Pass/g,"P").trim()
                        line = line.replace(/PASS/g,"P").trim()
                        line = line.replace(/NT/g,"N")
                        line = line.replace(/ =\d+=/g, '');
                        line = line.replace(/\s+/g,"-")
                        if (document.getElementById('biddingInput').value) {
                            document.getElementById('biddingInput').value += '-' + line;
                        } else {
                            document.getElementById('biddingInput').value = line;
                        }
                    }
                    if (tokenKey === 'Play') {
                        document.getElementById('playInput').value += (line.replace(/\s+/g, '') + " ").replace("*", "");
                    }
                } else {
                    tokenKey = ''
                }
                
            });
            // Get the checkbox element
            const checkbox = document.getElementById('format');
            
            // Set the checkbox to true (checked)
            checkbox.checked = true;

            // Optional: Update the data-default attribute if needed
            checkbox.setAttribute('data-default', 'true');
            //displayParsedData(parsedData);
        }

        function displayParsedData(parsedData) {
            const parsedDataDiv = document.getElementById('parsedData');
            parsedDataDiv.innerHTML = '';

            parsedData.forEach((item) => {
                parsedDataDiv.innerHTML += `${item.key}: ${item.value} <br>`;
            });
        }

        function readImportedFile() {
            // Get the file from the input field
            const fileInput = document.getElementById('importFile');
            const file = fileInput.files[0];

            // Check if a file was actually selected
            if (!file) {
                alert('Please select a file to import.');
                return;
            }

            // Read the file as text
            const reader = new FileReader();
            reader.onload = () => {
                const text = reader.result;
                const filteredLines = text.split('\n')
                    .filter((line) => !line.startsWith("%"))
                    .join('\n');
                const input = document.querySelector('#importInput');
                input.value = filteredLines;
            };

            // Start reading the file
            reader.readAsText(file);
        }

        function toggleSamples(containerId) {
            const sampleLines = document.getElementById(containerId);
            sampleLines.classList.toggle('hidden');
        }

        function showTooltip(idx) {
            var tooltip = document.getElementById("tooltip"+idx);
            tooltip.style.display = "inline";
        }

        function hideTooltip(idx) {
            var tooltip = document.getElementById("tooltip"+idx);
            tooltip.style.display = "none";
        }        

        function validate_suit(part) {
            return part === '' || /\d|[TJQKAX]/.test(part);
        }

        function validateHand(hand) {
            const suits = hand.split('.');
            if (suits.length !== 4) {
                console.log("Not 4 suits in ", hand);
                return false;
            }
            if (hand.length !== 16) {
                console.log("Not 13 cards ", hand);
                return false;
            }
            const result = suits.every(p => validate_suit(p.trim()));
            if (!result) {
                console.log("Wrong format ", hand);
                return false;
            }
            return true;
        }

        function validateBidding(bidding) {
            const regex = /^$|^([1-7][SHDCN]|X|XX|P)(-([1-7][SHDCN]|X|XX|P))*$/;
            return regex.test(bidding);
        }

        function validatePlay(play) {
            const regex = /^([SHDC][2-9TJQKA]){0,52}$/;
            return regex.test(play);
        }

        let bids = 0;
        function displayBid(data) {
            let alerted = "";
            if (data.alert == "True") {
                alerted = "*";
            }
            let explanation = "";
            if (data.explanation) {
                explanation = ` (${data.explanation})`;
   				explanation = explanation.replace(/!S/g, '<span style="color: blue">&spades;</span>');
				explanation = explanation.replace(/!H/g, '<span style="color: red">&hearts;</span>');
				explanation = explanation.replace(/!D/g, '<span style="color: orange">&diams;</span>');
				explanation = explanation.replace(/!C/g, '<span style="color: green">&clubs;</span>');
            }
            let html = `<br>
                <p class="bid"><strong>Bid:</strong> ${data.bid}${alerted} ${explanation} ${data.who !== undefined ? ' by ' + data.who : '' }</p>
                `;
            if (data.candidates && data.candidates.length > 0)
                html += `
                    <p><strong>Candidates:</strong>
                    <ul>
                        ${data.candidates.map(candidate => {
                            let bid = candidate.call.replace("PASS", "P"); // Replace only once
                            // Helper function to ensure numbers are 5 characters wide
                            const formatNumber = (num) => String(num).padStart(5, ' ');

                            return `
                                <li>
                                    Bid: ${bid.length === 1 ? '&nbsp;' + bid : bid}
                                    ${candidate.expected_score !== undefined ? `, Expected score: ${formatNumber(candidate.expected_score)}` : ''}
                                    ${candidate.expected_mp !== undefined ? `, Expected MP: ${formatNumber(candidate.expected_mp)}` : ''}
                                    ${candidate.expected_imp !== undefined ? `, Expected IMP: ${formatNumber(candidate.expected_imp)}` : ''}
                                    ${candidate.who !== undefined ? `, NN Score: ${candidate.who}&nbsp;&nbsp` : candidate.insta_score !== undefined ? `, NN Score: ${formatNumber(candidate.insta_score)}` : ''}
                                    ${candidate.adjustment !== undefined ? `, Adjusted: ${candidate.adjustment}` : ''}
                                </li>                               
                            `.replace(/\n\s+/g, ''); // Removes newlines + extra spaces;
                        }).join('')}
                    </ul></p>
                    `;
            if (data.hcp && (data.hcp != -1)) {
                let formattedShape = [];
                for (let i = 0; i < data.shape.length; i += 4) {
                    formattedShape.push(data.shape.slice(i, i + 4).join('-'));
                }
                formattedShape = formattedShape.join(' &nbsp;&nbsp; ');                                    
                html += `
                    <p><strong>HCP:</strong> ${data.hcp.join(' - ')}</p>
                    <p><strong>Shape:</strong> ${formattedShape}</p>
                    `;
                if (data.quality != null) {
                    html += `<br><p><strong>Sample quality: </strong>${data.quality}</p>
                    `;
                }
            }
            if ("samples" in data) {
                html += generateSamplesTable(data.samples, bids);
                bids += 1;
            }
            document.querySelector('#info').innerHTML = html + document.querySelector('#info').innerHTML

        }

        function displayPlay(data, player, declarer, actual_hand) {
            if (actual_hand == 1) {
                // We lead from dummy
                player = (player + 2) % 4
            }
            let html = "<h3>Play</h3><br> " + data["card"] +  (data.who !== undefined ? " selected by " + data["who"] : "")  + "<br/><br/>"

            if ("candidates" in data && data.candidates.length > 0) {
                html += '<h3>Candidates</h3>'
                html += '<p>We have 3 different parameters for selecting the card, and first goal is to make/set the contract (if not matchpoints), then the double dummy score, and finally the score from the neural network. '
                html += 'If the quality of the samples are bad (or the nn suggest a specific card with confidence), then we select the that card. '
                html += 'Also be aware that the data is rounded to nearest even number before comparing.</p>'
                html += 'OR use MP or IMP calculation instead of tricks.<br><br>'
                html += '<table>'

                for (const candidate of data.candidates) {

                    html += '<tr>'
                    html += '<td class="candidate-card">' + candidate['card'] + '</td>'
                    if ("expected_tricks_sd" in candidate) {
                        html += '<td>e(tricks)(SD)=' + Math.round(candidate['expected_tricks_sd'] * 100) / 100 + '</td>'
                    }
                    if ("expected_tricks_dd" in candidate) {
                        html += '<td>e(tricks)(DD)=' + Math.round(candidate['expected_tricks_dd'] * 100) / 100 + '</td>'
                    }
                    if ("expected_score_sd" in candidate) {
                        html += '<td>e(score)sd=' + Math.round(candidate['expected_score_sd'] * 100) / 100 + '</td>'
                    }
                    if ("expected_score_dd" in candidate) {
                        html += '<td>e(score)dd=' + Math.round(candidate['expected_score_dd'] * 100) / 100 + '</td>'
                    }
                    if ("expected_score_mp" in candidate) {
                        html += '<td>e(MP)dd=' + Math.round(candidate['expected_score_mp'] * 100) / 100 + '%</td>'
                    }
                    if ("expected_score_imp" in candidate) {
                        html += '<td>e(IMP)dd=' + candidate['expected_score_imp'] + '</td>'
                    }
                    if ("insta_score" in candidate) {
                        html += '<td>iscore=' + Math.round(candidate['insta_score'] * 1000) / 1000 + '</td>'
                    }
                    if ("p_make_contract" in candidate) {
                        html += '<td>e(make/set)=' + Math.round(candidate['p_make_contract'] * 100) / 100 + '</td>'
                    }
                    if ("msg" in candidate) {
                        if (candidate['msg'].length > 0) {
                            var escapedMsg = candidate['msg']
                            .replace(/ /g, '&nbsp;')  // Escape space
                            html += "<td onmouseover=BENShowPopup(this,'" + escapedMsg + "',0) onmouseout=BENHidePopup()>[calculations]</td>"
                        }
                    }
                    html += '</tr>'
                }

                html += '</table><br>'
            }

            if ("hcp" in data && "shape" in data) {
                if (data['hcp'] != -1 && data['shape'] != -1) {
                    let shape = data['shape'].reduce((acc, val) => acc.concat(val), []);
                    html += '<h3>Bidding Info</h3>'
                    if (data['hcp'].length > 2) {
                        html += '<div>Dummy: ' + data['hcp'][0] + ' hcp, shape: '
                        for (let i = 0; i < 4; i++) {
                            html += shape[i] + " "
                        }
                        html += '</div>'
                        html += '<div>Partner: ' + data['hcp'][1] + ' hcp, shape: '
                        for (let i = 0; i < 4; i++) {
                            html += shape[i + 4] + " "
                        }
                        html += '</div>'
                        html += '<div>Declarer: ' + data['hcp'][2] + ' hcp, shape: '
                        for (let i = 0; i < 4; i++) {
                            html += shape[i + 8] + " "
                        }
                        html += '</div>'
                    } else {
                        // we are seated after declarer
                        if ((player - declarer + 4) % 4 == 1) {
                            html += '<div>Declarer: ' + data['hcp'][0] + ' hcp, shape: '
                            for (let i = 0; i < 4; i++) {
                                html += shape[i] + " "
                            }
                            html += '</div>'
                            html += '<div>Partner: ' + data['hcp'][1] + ' hcp, shape: '
                            for (let i = 0; i < 4; i++) {
                                html += shape[i + 4] + " "
                            }
                            html += '</div>'
                        } else
                            // we are seated before declarer
                            if ((player - declarer + 4) % 4 == 3) {
                                html += '<div>Partner: ' + data['hcp'][0] + ' hcp, shape: '
                                for (let i = 0; i < 4; i++) {
                                    html += shape[i] + " "
                                }
                                html += '</div>'
                                html += '<div>Declarer: ' + data['hcp'][1] + ' hcp, shape: '
                                for (let i = 0; i < 4; i++) {
                                    html += shape[i + 4] + " "
                                }
                                html += '</div>'
                            }
                            else {
                                // RHO
                                if (player == 0) {
                                    html += '<div>West: ' + data['hcp'][0] + ' hcp, shape: '
                                } 
                                if (player == 1) {
                                    html += '<div>North: ' + data['hcp'][0] + ' hcp, shape: '
                                } 
                                if (player == 2) {
                                    html += '<div>East: ' + data['hcp'][0] + ' hcp, shape: '
                                } 
                                if (player == 3) {
                                    html += '<div>South: ' + data['hcp'][0] + ' hcp, shape: '
                                } 
                                
                                for (let i = 0; i < 4; i++) {
                                    html += shape[i] + " "
                                }
                                html += '</div>'
                                // LHO
                                if (player == 0) {
                                    html += '<div>East: ' + data['hcp'][1] + ' hcp, shape: '
                                } 
                                if (player == 1) {
                                    html += '<div>South: ' + data['hcp'][1] + ' hcp, shape: '
                                } 
                                if (player == 2) {
                                    html += '<div>West: ' + data['hcp'][1] + ' hcp, shape: '
                                } 
                                if (player == 3) {
                                    html += '<div>North: ' + data['hcp'][1] + ' hcp, shape: '
                                } 
                                for (let i = 0; i < 4; i++) {
                                    html += shape[i + 4] + " "
                                }
                                html += '</div>'
                            }
                    }
                }

            }
            if (data.quality != null) {
                    html += `
                    <br><p><strong>Sample quality: </strong>${data.quality}</p>
                    `;
                }

            if ("samples" in data) {
                html += generateSamplesTable(data.samples, bids);
                bids += 1;
            }

            document.querySelector('#info').innerHTML = html + document.querySelector('#info').innerHTML
        }

        // Function to load user value from localStorage
        function loadUser() {
            const savedUser = localStorage.getItem('user');
            if (savedUser) {
                document.getElementById('userInput').value = savedUser;
            }
        }

        // Function to save user value to localStorage
        function saveUser() {
            const user = document.getElementById('userInput').value;
            localStorage.setItem('user', user);
        }

        function findDeclarer(bidding) {
            contract = bidding.substring(bidding.length - 8, bidding.length - 6);
            firstbid = bidding.indexOf(contract[1]);
            const dealer = "NESW".indexOf(document.getElementById('dealerInput').value);
            declarer = ((firstbid + 1) / 2 + dealer - 1) % 4
            console.log(declarer, contract, firstbid, dealer);
            return declarer
        }

        // Call loadUser() when the page loads
        window.onload = loadUser;

        async function callAPI(action) {
            document.querySelector("#loader").style.visibility = "visible"; 
            let validationerror = false;
            let hand = document.getElementById('handInput').value.toUpperCase();
            if (!validateHand(hand) && (action != "explain") && (action != "explain_auction")) {
                alert("Invalid hand input. Please enter four suits delimited by .");
                validationerror = true;
            }

            let bidding = document.getElementById('biddingInput').value.toUpperCase();
            if (!validateBidding(bidding)) {
                alert("Invalid bidding input. Please enter the bids separated by hyphens. Allowed bids: 1C, 1D, 1H, 1S, 1N, ..., 7N, X, XX, P");
                validationerror = true;
            }

            let played =  document.getElementById('playInput').value.toUpperCase();
            played = played.replace(/\s/g,"")

            if (!validatePlay(played)) {
                alert("Invalid played input. Please enter the cards like this: S2SKS4");
                validationerror = true;
            }

            if ((played == "") && (action == "play")){
                action = "lead";
            }

            let dummy = document.getElementById('dummyInput').value.toUpperCase();
            if ((action == 'play') && !validateHand(dummy)) {
                alert("Invalid dummy input. Please enter four suits delimited by .");
                validationerror = true;
            }

            if ((action == 'contract') && !validateHand(dummy)) {
                alert("Invalid dummy input. Please enter four suits delimited by .");
                validationerror = true;
            }

            if (validationerror) {
                document.querySelector("#loader").style.visibility = "hidden";
                return;
            }

            bidding = bidding.replace(/XX/g, 'Rd');
            bidding = bidding.replace(/X/g, 'Db');
            bidding = bidding.replace(/-/g, '');
            // Replace 'P' with '--'
            bidding = bidding.replace(/P/g, '--');

            const user = document.getElementById('userInput').value;
            const dealer = document.getElementById('dealerInput').value;
            const seat = document.getElementById('seatInput').value;
            const player = "NESW".indexOf(seat)
            const vul = document.getElementById('vulInput').value;
            const mp = document.getElementById('matchpoint').checked;
            const explain = document.getElementById('explain').checked;
            const format = document.getElementById('format').checked;
            const matchpoint = mp  ?  "mp" : "imp";
            // Get the current hostname and protocol
            const hostname = window.location.hostname;
            const protocol = window.location.protocol;
            const old_server = document.getElementById('old_server').checked;
            const details = document.getElementById('details').checked;
            let port = 8085
            if (old_server) {
                port = 8088
            }

            var url = `${protocol}//${hostname}:${port}/${action}?user=${user}&dealer=${dealer}&seat=${seat}&vul=${vul}&ctx=${bidding}&hand=${hand}&tournament=${matchpoint}&explain=${explain}&format=${format}&details=${details}`;
            if (action == "play" || action == "claim")
                url += `&dummy=${dummy}&played=${played}`;
            if (action == "contract")
                url += `&dummy=${dummy}`;

            try {           
                const response = await fetch(url, {
                    method: 'GET'
                })

                document.querySelector("#loader").style.visibility = "hidden";
                // Check if the response is successful
                if (!response.ok) {
                    // Log the response status and status text
                    console.error('Response not OK:', response.status, response.statusText);

                    // Parse the response body as JSON
                    const errorResponse = await response.json();
                    
                    // Extract the error message from the JSON response
                    const errorMessage = errorResponse.error || 'Unknown error occurred';

                    // Show the error message to the user
                    alert(errorMessage);
                    return;
                }

                // Parse the response as JSON
                const data = await response.json();
                if (data.message) {
                    document.querySelector("#loader").style.visibility = "hidden";
                    alert(data.message)
                    return
                }

                if (action == 'bid') {
                    if (data.alert == "True") {
                        document.getElementById('result').innerText = `BEN Suggest: ${data.bid}*`;
                    } else {
                        document.getElementById('result').innerText = `BEN Suggest: ${data.bid}`;
                    }
                    displayBid(data);
                }
                if (action == 'contract') {
                    document.getElementById('result').innerText = 'BEN Suggest:';
                    for (var key in data) {
                        if (data.hasOwnProperty(key)) {
                            var entry = data[key];
                            var div = document.createElement('div');
                            // Ensure Tricks and Percentages are arrays of the same length
                            var tricks = Array.isArray(entry.Tricks) ? entry.Tricks : [];
                            var percentages = Array.isArray(entry.Percentage) ? entry.Percentage : [];

                            // Pair tricks with percentages, then sort by percentage descending
                            var tricksWithPercentages = tricks.map((trick, index) => {
                                return { trick: trick, percentage: percentages[index] };
                            }).sort((a, b) => b.percentage - a.percentage); // Sort by percentage descending

                            // Format tricks with percentages
                            var tricksFormatted = tricksWithPercentages.map(item => {
                                return `${item.trick} (${item.percentage})`; // Explicitly format as string
                            }).join(', ');

                            div.innerHTML = `
                                Contract: ${key} (Score: ${entry.score} Tricks: ${tricksFormatted})
                            `;
                            document.getElementById('result').appendChild(div);
                        }
                    }
                }
                if (action == 'explain' || action == 'explain_auction') {
                    explanation = `${data.explanation}`;
                    explanation = explanation.replace(/!S/g, '<span style="color: blue">&spades;</span>');
                    explanation = explanation.replace(/!H/g, '<span style="color: red">&hearts;</span>');
                    explanation = explanation.replace(/!D/g, '<span style="color: orange">&diams;</span>');
                    explanation = explanation.replace(/!C/g, '<span style="color: green">&clubs;</span>');
                    document.getElementById('result').innerHTML = `Meaning: ${explanation}`;
                }
                if (action == 'claim') {
                    document.getElementById('result').innerText = `Response: ${data.result}`;
                }
                if ((action == 'play') || (action == 'lead')) {
                    const declarer = findDeclarer(bidding)
                    document.getElementById('result').innerText = `BEN Suggest: ${data.card}`;
                    displayPlay(data, player, declarer, data["player"]);
                }
            } catch (error) {
                document.querySelector("#loader").style.visibility = "hidden";
                // Handle any errors that occur during the fetch request
                alert('Error fetching data:' + error.message);
                // Show an error message to the user or perform other error handling actions
            }
        }
        document.querySelector("#loader").style.visibility = "hidden";
        function BENGetXY(obj) {
            var pt = {x: 0, y: 0};
            var div = document.getElementById("bccontent");
            if (div)
                pt.y = -div.offsetTop;

            while (obj) {
                pt.x += obj.offsetLeft;
                pt.y += obj.offsetTop;
                obj = obj.offsetParent;
            }
            return pt;
        }
        function BENShowPopup(obj, text, pos) {
            var div = document.getElementById("BENPopup");
            text = text.replace(/\&/g,"&amp;");
            text = text.replace(/\</g,"&lt;");
            div.innerHTML = text.replace(/\|/g,"<br/>");
            var pt = BENGetXY(obj);
            switch (pos) {
            case 0:		// right
                div.style.left = pt.x + obj.offsetWidth + 8 + "px";
                div.style.top = pt.y + "px";
                break;
            case 1:		// above
                div.style.left = pt.x + 8 + "px";
                div.style.top = pt.y - div.offsetHeight + "px";
                break;
            case 2:		// left
                var x = pt.x - div.offsetWidth;
                if (x < 0) x = 0;
                div.style.left = x + "px";
                div.style.top = pt.y + "px";
                break;
            case 3:		// below
                div.style.left = pt.x + 8 + "px";
                div.style.top = pt.y + obj.offsetHeight + "px";
                break;
            }
            div.style.visibility = "visible";
        }
        function BENHidePopup() {
            var div = document.getElementById("BENPopup");
            div.style.visibility = "hidden";
        }
    </script>
    <div id=BENPopup style="position: absolute; color: black; background-color: #ffffe1; border: thin solid black; text-align: left; padding-left: 4px; padding-right: 4px; font: small 'Arial', sans-serif; visibility: hidden"></div>
</body>

</html>