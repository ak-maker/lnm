import json
import random

def test_format(instruction_answer_pairs):
    # Print the extracted instruction-answer pairs
    formatted_data = []
    for instruction in instruction_answer_pairs:
        # print(f"Instruction: {instruction}")
        # print(f"Answer: {answer}")
        # print('---')
        user_message = {
            "role": "user",
            "content": f"Answer the following question about Sionna: {instruction}"
        }
        entry = {
            "messages": [system_message, user_message]
        }
        formatted_data.append(entry)
    return formatted_data




file_path = 'E:\python\lnm\\automation_for_markdown\\finetune_prepare\\test.md'

current_instruction = None


# Initialize a list to hold the extracted instruction-answer pairs
instruction_pairs = []

# Read the file line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line starts with "INSTRUCTION:"
        if line.startswith("Instruction: "):
            # Check if the next line starts with "ANSWER:"
            next_line = next(file, None)
            if next_line is None or not next_line.startswith("Answer: "):
                raise ValueError("Missing 'ANSWER:' after 'INSTRUCTION:'")
            instruction_pairs.append(line.split("Instruction: ")[1])


system_message = {
        "role": "system",
        "content": "You are an expert in Sionna, capable of writing code and illustrating information like formulas and APIs. Your proficiency in Sionna allows you to simulate and analyze various aspects of wireless communication systems effectively."
}

# test_data = test_format(instruction_pairs)



with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\test_data.md', 'w', encoding='utf-8') as f:
    for entry in instruction_pairs:
        f.write(entry)


