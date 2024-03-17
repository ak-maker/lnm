import json

# The path to your .jsonl file on your local machine
jsonl_file_path = 'E:\\python\\lnm\\automation_for_markdown\\test_results.jsonl'
data = []

# Read the .jsonl file
with open(jsonl_file_path, 'r') as file:
    for line in file:
        # Convert each line to a JSON object
        json_data = json.loads(line)
        data.append(json_data)

# Sort the data based on the first element in each list
sorted_data = sorted(data, key=lambda x: x[0])

# The path where you want to save the new sorted .jsonl file
sorted_jsonl_file_path = 'E:\\python\\lnm\\automation_for_markdown\\test_results.jsonl'

# Write the sorted data back to a new .jsonl file
with open(sorted_jsonl_file_path, 'w') as file:
    for entry in sorted_data:
        # Convert the JSON object to a string and write it to the file
        file.write(json.dumps(entry) + '\n')
