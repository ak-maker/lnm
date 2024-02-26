import os
import re

def count_instructions_in_file(file_path):
    """Count the number of 'INSTRUCTION:' occurrences in a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Use a regular expression to find all occurrences of '### <number>. INSTRUCTION:'
        instructions = re.findall(r'### \d+\. INSTRUCTION:', content)
        return len(instructions)

# Path to the directory containing files
directory_path = '../../success_files'

# Iterate over each file in the directory and count the instructions
total_instructions = 0
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        num_instructions = count_instructions_in_file(file_path)
        total_instructions += num_instructions
        print(f"File '{filename}' contains {num_instructions} instructions.")

# Print the total count of instructions
print(f"Total instructions in all files: {total_instructions}")
