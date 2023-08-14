import sys

def generate_html_card(suit, cards):
    html = f"<div class='suit'><span>{suit}</span>"
    for card in cards:
        html += f"{card}"
    html += "</div>"
    return html

def generate_html_deal(line, board_number):
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
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=0"> Se it played (no search for NS) </a><br>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=1"> Se it played (no search for EW) </a><br>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=4"> Se it played (Search for both) </a><br>
        <div id="auction"></div>

    </body>
    </html>"""

    filename = f"./{folder}/Board{board_number}.html"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html)


if len(sys.argv) < 3:
    print("Usage: python printdeal.py filename folder")
    sys.exit(1)

filename = sys.argv[1]
folder = sys.argv[2]

# Read the file and generate HTML for each line
with open(filename, 'r') as file:
    for board_number, line in enumerate(file, start=1):
        generate_html_deal(line.strip(), board_number)
