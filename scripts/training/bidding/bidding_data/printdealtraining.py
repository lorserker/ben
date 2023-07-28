import itertools

def generate_html_card(suit, cards):
    html = f"<div class='suit'><span>{suit}</span>"
    for card in cards:
        html += f"{card}"
    html += "</div>"
    return html

def generate_html_deal(line, board_number, bidding):
    parts = line.split()
    dealer = parts[0]
    vulnerable = parts[1]
    cards = parts[2:]

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='utf-8'>
        <title>Match deal</title>
        <link rel='stylesheet' href='viz.css'>
        <script type="text/javascript" src="viz.js"></script>  
    </head>
    <body>
        <div id='deal'>
            <div id='dealer-vuln'>
                <div id='vul-north' class='{"red" if vulnerable in ('N-S', 'Both') else 'white'}'>
                    {"<span class='dealer'>N</span>" if dealer == 'N' else ''}
                </div>
                <div id='vul-east' class='{"red" if vulnerable in ('E-W', 'Both') else 'white'}'>
                    {"<span class='dealer'>E</span>" if dealer == 'E' else ''}
                </div>
                <div id='vul-south' class='{"red" if vulnerable in ('N-S', 'Both') else 'white'}'>
                    {"<span class='dealer'>S</span>" if dealer == 'S' else ''}
                </div>
                <div id='vul-west' class='{"red" if vulnerable in ('E-W', 'Both') else 'white'}'>
                    {"<span class='dealer'>W</span>" if dealer == 'W' else ''}
                </div>
                <div id='board'>
                    {board_number}
                </div>
            </div>
            <div id='north'>
                {generate_html_card('&spades;', cards[0].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[0].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[0].split('.')[2])}
                {generate_html_card('&clubs;', cards[0].split('.')[3])}
            </div>
            <div id='west'>
                {generate_html_card('&spades;', cards[3].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[3].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[3].split('.')[2])}
                {generate_html_card('&clubs;', cards[3].split('.')[3])}
            </div>
            <div id='east'>
                {generate_html_card('&spades;', cards[1].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[1].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[1].split('.')[2])}
                {generate_html_card('&clubs;', cards[1].split('.')[3])}
            </div>
            <div id='south'>
                {generate_html_card('&spades;', cards[2].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[2].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[2].split('.')[2])}
                {generate_html_card('&clubs;', cards[2].split('.')[3])}
            </div>
        </div>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=0"> Se it played </a><br>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=1"> Se it played </a>
        <div id="auction">{bidding}</div>

        <script type="text/javascript">
            let auction = new Auction({'NESW'.index(dealer)}, {bidding})
            auction.render(document.getElementById("auction"))
        </script>
    </body>
    </html>"""

    filename = f"./training/html/BoardX{board_number}.html"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html)


# Read the file and generate HTML for each line
with open('D:/github/ben/scripts/training/bidding/bidding_data/bidding_data.txt', 'r') as file:
    #lines = list(itertools.islice(file, 20))
    lines = list(file)
    #print(lines)
    for board_number, (line1, line2) in enumerate(zip(lines[::2], lines[1::2]), start=1):
        #print(f"{' '.join(line2.split()[:2])} {line1}")
        bidding = [{'bid': bid} for bid in line2.split()[2:]]
        for bid_dict in bidding:
            if bid_dict['bid'] == 'P':
                bid_dict['bid'] = 'PASS'
        #print(bidding)
        if (line2 == "W None 1N 2D P 2S P P 3D P 3H P 4S P 4N P 5N P P P\n"):
            generate_html_deal(f"{' '.join(line2.split()[:2])} {line1}", board_number, bidding)
        #else:
            #print(line2 + "X")
