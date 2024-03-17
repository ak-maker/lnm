import os

def find_files_with_prefix(directory, prefix):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                matched_files.append(os.path.join(root, file))
    return matched_files

def count_phrases(files, phrases):
    phrase_counts = {phrase: 0 for phrase in phrases}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                for phrase in phrases:
                    if phrase in line:
                        phrase_counts[phrase] += 1
    return phrase_counts

def find_unmatched_instruction(files):
    unmatched_instructions = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith('INSTRUCTION: '):
                    # 检查是否存在'ANSWER:'在紧接的下一行
                    if i+1 < len(lines) and not lines[i+1].startswith('ANSWER:'):
                        unmatched_instructions.append((file_path, i+1))
    return unmatched_instructions

# 使用示例
directory = 'E:\python\lnm\\automation_for_markdown\data_chunks'
prefix = 'IA_'
files = find_files_with_prefix(directory, prefix)
phrases = ['INSTRUCTION: ', 'ANSWER:']
counts = count_phrases(files, phrases)
print(counts)
unmatched = find_unmatched_instruction(files)
for i in unmatched:
    print(i)