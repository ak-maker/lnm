# find the instructions
with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\test_data.md', 'r') as file:
    instructions = [line for line in file]


# Read the file and extract lines
with open('E:\python\lnm\\automation_for_markdown\\test_results.jsonl', 'r') as file:
    lines = file.readlines()

# Convert each line to a list and sort by the first element
sorted_lines = sorted([eval(line.strip()) for line in lines], key=lambda x: x[0])

# Store the sorted lines in a dictionary
sorted_dict = {line[0]: line[1] for line in sorted_lines}



# Print the sorted dictionary
with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\gpt4_score_shown.md', 'w', encoding='utf-8') as f:
    i = 0
    for value in sorted_dict.values():
        value = value.split("\n\n")
        f.write(f"Instruction: {instructions[i]}")
        i = i + 1
        total_score = float(value[0].split("\n")[0].split("Total Score: ")[1])
        total_score = round((total_score*10.0)/24.0, 1)
        seperated_score = value[0].split("\n")[1]
        f.write(f"Total Score: {total_score}"+"\n")
        f.write(seperated_score+"\n")
        value[1] = value[1].replace(".", ".\n")
        f.write(f"Reason: {value[1]}"+"\n\n")

