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

        # Append the extracted values to the list as a tuple if imp is not 0
        if imp != 0:
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
sorted_data = sorted(data_list, key=lambda x: x[7], reverse=True)

# Print the sorted output
for board_data in sorted_data:
    board, contract1, dd_tricks1, dd_score1, contract2, dd_tricks2, dd_score2, imp = board_data
    output = f"Board {board:>3}: {contract1:>5}, {dd_tricks1:>3}, {dd_score1:>5}, {contract2:>5}, {dd_tricks2:>3}, {dd_score2:>5}, {imp:>5}"
    print(output)

# Print the sum of positive and negative imp values
print(f"Positive Imp Sum: {positive_imp_sum}")
print(f"Negative Imp Sum: {negative_imp_sum}")

# Determine the winner or draw based on the sum of imp values
total_imp_sum = positive_imp_sum + negative_imp_sum
if total_imp_sum > 0:
    print(f"NS wins by {total_imp_sum}")
elif total_imp_sum < 0:
    print(f"EW wins by {abs(total_imp_sum)}")
else:
    print("It is a draw")
