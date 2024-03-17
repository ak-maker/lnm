import json
import random

def train_and_validate_format(instruction_answer_pairs):
    # Print the extracted instruction-answer pairs
    formatted_data = []
    for instruction, answer in instruction_answer_pairs:
        # print(f"Instruction: {instruction}")
        # print(f"Answer: {answer}")
        # print('---')
        user_message = {
            "role": "user",
            "content": f"{instruction}"
        }
        assistant_message = {
            "role": "assistant",
            "content": f"{answer}"
        }
        entry = {
            "messages": [system_message, user_message, assistant_message]
        }
        formatted_data.append(entry)
    return formatted_data




file_path = 'E:\python\lnm\\automation_for_markdown\\finetune_prepare\IA_total.md'



current_instruction = None
current_answer = []

# Initialize a list to hold the extracted instruction-answer pairs
instruction_answer_pairs = []

# Read the file line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line starts with "INSTRUCTION:"
        if line.startswith("INSTRUCTION:"):
            # Check if the next line starts with "ANSWER:"
            next_line = next(file, None)
            if next_line is None or not next_line.startswith("ANSWER:"):
                raise ValueError("Missing 'ANSWER:' after 'INSTRUCTION:'")
            # Save the current instruction and answer pair if they exist
            if current_instruction and current_answer:
                # Join the answer lines into a single string
                answer_str = ''.join(current_answer).rstrip('\n')
                instruction_answer_pairs.append((current_instruction, answer_str))
                # Reset the current answer for the next pair
                current_answer = []
            # Update the current instruction
            current_instruction = line[len("INSTRUCTION:"):].strip()
            # Start collecting the answer
            current_answer.append(next_line[len("ANSWER:"):])
        elif current_instruction is not None:
            # Accumulate lines for the current answer
            current_answer.append(line)

# Save the last instruction and answer pair if they exist
if current_instruction and current_answer:
    answer_str = ''.join(current_answer).rstrip('\n')
    instruction_answer_pairs.append((current_instruction, answer_str))

system_message = {
        "role": "system",
        "content": "You are an expert in Sionna, capable of writing code and illustrating information like formulas and APIs. Your proficiency in Sionna allows you to simulate and analyze various aspects of wireless communication systems effectively."
}
# random 50 test
random.shuffle(instruction_answer_pairs)
selected_pairs = instruction_answer_pairs[:50]
instruction_answer_pairs = instruction_answer_pairs[50:]
with open('E:\\python\\lnm\\automation_for_markdown\\finetune_prepare\\test.md', 'w', encoding='utf-8') as file:
    for instruction, answer in selected_pairs:
        file.write(f"Instruction: {instruction}\n")
        file.write(f"Answer: {answer}\n")
        file.write('---\n')


# validation and traning
split_index = int(len(instruction_answer_pairs) * 0.9) + 1

# 随机打乱剩余的问答对
random.shuffle(instruction_answer_pairs)

# 分割为训练和验证集
training_pairs = instruction_answer_pairs[:split_index]
training_json = train_and_validate_format(training_pairs)
validation_pairs = instruction_answer_pairs[split_index:]
validation_json = train_and_validate_format(validation_pairs)

# 将训练集写入训练文件
with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\train_data.jsonl', 'w', encoding='utf-8') as f:
    for entry in training_json:
        f.write(json.dumps(entry))
        f.write("\n")



# 将验证集写入验证文件
with open('E:\\python\\lnm\\automation_for_markdown\\finetune_prepare\\validate_data.jsonl', 'w', encoding='utf-8') as fp:
    for entry in validation_json:
        fp.write(json.dumps(entry))
        fp.write("\n")