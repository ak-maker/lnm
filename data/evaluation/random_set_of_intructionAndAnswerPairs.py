import os
import random
import re

def extract_instruction_answer_pairs(file_path):
    """Extract instruction-answer pairs from a file."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Split the content by '###' to separate each pair
        sections = content.split('###')[1:]  # Skip the first split as it will be empty
        for section in sections:
            # Extract instruction and answer using regular expressions
            instruction_match = re.search(r'INSTRUCTION:\s*(.*?)\s*\*\*ANSWER:\*\*', section, re.DOTALL)
            answer_match = re.search(r'\*\*ANSWER:\*\*\s*(.*?)$', section, re.DOTALL)
            if instruction_match and answer_match:
                instruction = instruction_match.group(1).strip()
                answer = answer_match.group(1).strip()
                pairs.append((instruction, answer))
    return pairs

def random_instruction_answer_pairs(pairs, num_samples):
    """Randomly select a set of instruction-answer pairs."""
    return random.sample(pairs, num_samples)

# Path to the directory containing files with instruction-answer pairs
directory_path = '../../success_files'

# Extract instruction-answer pairs from all files in the directory
all_pairs = []
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        pairs = extract_instruction_answer_pairs(file_path)
        all_pairs.extend(pairs)
print(len(all_pairs))
# Generate a random set of instruction-answer pairs
num_samples = 5  # Number of pairs to randomly select
random_pairs = random_instruction_answer_pairs(all_pairs, num_samples)

# Print the randomly selected instruction-answer pairs
for i, (instruction, answer) in enumerate(random_pairs, start=1):
    print(f"### {i}. INSTRUCTION:\n{instruction}\n\n**ANSWER:**\n\n{answer}\n")
