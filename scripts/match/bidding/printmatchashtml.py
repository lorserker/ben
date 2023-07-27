import json
import sys

# Read the file line by line and process each JSON object
data_list = []
positive_imp_sum = 0
negative_imp_sum = 0

for board, line in enumerate(sys.stdin, start=1):
    # Remove leading/trailing whitespaces and newline characters
    line = line.strip()

    try:
        # Parse the JSON data
        data = json.loads(line)

        # Extract the required values
        contract = data['contract']
        dd_tricks = data['dd_tricks']
        dd_score = data['dd_score']
        imp = data['imp']

        # Process contract parts and replace None values with "Pass"
        contract_parts = []
        for part in contract:
            if part is None:
                contract_parts.append("Pass")
            else:
                contract_parts.append(part)

        data_list.append((board, contract_parts[0], dd_tricks[0], dd_score[0], contract_parts[1], dd_tricks[1], dd_score[1], imp))

        # Sum positive and negative imp values
        if imp > 0:
            positive_imp_sum += imp
        else:
            negative_imp_sum += imp

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on Board {board}: {e}")

    except KeyError as e:
        print(f"Error extracting data from JSON on Board {board}: {e}")

# Sort the data_list based on the imp value in descending order
sorted_data = sorted(data_list, key=lambda x: x[0], reverse=False)

# Generate the HTML tables
table1_html = "<table class='border-collapse table-container'>"
table1_html += "<style>"
table1_html += "th { background-color: #4a86e8; color: white; }"
table1_html += ".align-right { text-align: right; }"
table1_html += ".positive-imp { background-color: #b8e994; }"
table1_html += ".negative-imp { background-color: #ff7675; }"
table1_html += ".zero-imp { background-color: white; }"
table1_html += ".align-center { text-align: center; }"
table1_html += ".row-height { height: 22px; }"
table1_html += "</style>"
table1_html += "<tr><th>Board</th><th>Contract</th><th>Tricks</th><th>Result</th><th>Contract</th><th>Tricks</th><th>Result</th><th class='align-right'>Imps (+)</th><th class='align-right'>Imps (-)</th></tr>"

table2_html = "<table class='border-collapse table-container'>"
table2_html += "<style>"
table2_html += "th { background-color: #4a86e8; color: white; }"
table2_html += ".align-right { text-align: right; }"
table2_html += ".positive-imp { background-color: #b8e994; }"
table2_html += ".negative-imp { background-color: #ff7675; }"
table2_html += ".zero-imp { background-color: white; }"
table2_html += ".align-center { text-align: center; }"
table2_html += ".row-height { height: 22px; }"
table2_html += "</style>"
table2_html += "<tr><th>Board</th><th>Contract</th><th>Tricks</th><th>Result</th><th>Contract</th><th>Tricks</th><th>Result</th><th class='align-right'>Imps (+)</th><th class='align-right'>Imps (-)</th></tr>"

for i, board_data in enumerate(sorted_data):
    board, contract1, dd_tricks1, dd_score1, contract2, dd_tricks2, dd_score2, imp = board_data

    # Align right for Result and Tricks columns
    result1 = f"<td class='align-right'>{dd_score1}</td>" if dd_score1 is not None else "<td class='align-right'></td>"
    tricks1 = f"<td class='align-right'>{dd_tricks1}</td>" if dd_tricks1 is not None else "<td class='align-right'></td>"
    result2 = f"<td class='align-right'>{dd_score2}</td>" if dd_score2 is not None else "<td class='align-right'></td>"
    tricks2 = f"<td class='align-right'>{dd_tricks2}</td>" if dd_tricks2 is not None else "<td class='align-right'></td>"

    # Split Imps column into positive and negative columns
    imp_positive = f"<td class='align-right'>{imp if imp > 0 else '--'}</td>"
    imp_negative = f"<td class='align-right'>{abs(imp) if imp < 0 else '--'}</td>"

    # Add class to the row based on imp value
    row_class = "positive-imp" if imp > 0 else "negative-imp" if imp < 0 else "zero-imp"
    row_height_class = "row-height"
    row_html = f"<tr class='{row_class} {row_height_class}'><td class='align-center'><a href='board{board}.html'>{board}</a></td><td>{contract1}</td>{tricks1}{result1}<td>{contract2}</td>{tricks2}{result2}{imp_positive}{imp_negative}</tr>\n"

    # Split rows evenly between the two tables
    if i < len(sorted_data) / 2:
        table1_html += row_html
    else:
        table2_html += row_html

table1_html += "</table>"
table2_html += "</table>"

# Print the winner at the top of the tables with <h1> tags
total_imp_sum = positive_imp_sum + negative_imp_sum
if total_imp_sum > 0:
    win_html = f"<h1>NS wins by {total_imp_sum}</h1>\n"
elif total_imp_sum < 0:
    win_html = f"<h1>EW wins by {abs(total_imp_sum)}</h1>\n"
else:
    win_html = "<h1>It is a draw</h1>\n"

# Print the tables and final score
print(" <!DOCTYPE html>")
print("    <html lang = 'en'>")
print("    <head>")
print("    <meta charset = 'utf-8'>")
print(" <title> Match deal </title>")
print("    <link rel = 'stylesheet' href = 'viz.css' >")
print(" </head>")
print(" <body>")

print("<div style='display: flex; justify-content: center;'>")
print("<p><h1 style='text-align: center;'>")
print(win_html)
print("</h1></p>")
print("</div>")
print("<div style='display: flex; justify-content: center;'>")
print("<div style='margin-right: 20px;'>")
print(table1_html)
print("</div>")
print("<div>")
print(table2_html)
print("</div>")
print("</div>")
print(f"<p style='text-align: center;'><b>Final score:</b> NS: {positive_imp_sum}, EW: {abs(negative_imp_sum)}</p>")
print("</body>     </html>")
