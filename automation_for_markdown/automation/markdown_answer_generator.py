import ast
import json
import markdown_prompt
import os
import subprocess
import re

# def safe_write(writer, text, encoding='utf-8'):
#     try:
#         writer.write(text)
#     except UnicodeEncodeError:
#         if encoding == 'utf-16':
#             # 如果已经是 utf-16 还失败，则放弃写入
#             print(f"Cannot encode text: {text}")
#         else:
#             # 如果 utf-8 写入失败，尝试 utf-16
#             safe_write(writer, text, 'utf-16')
def safe_write(path, text):
    try:
        # 尝试以utf-8编码写入
        with open(path, 'a', encoding='utf-8') as writer:
            writer.write(text)
    except UnicodeEncodeError:
        # 如果出错，打印错误信息，并以utf-16编码写入
        print("Encoding error encountered. Switching to utf-16 for the text")
        with open(path, 'a', encoding='utf-16') as writer:
            writer.write(text)

def extract_number(filename):
    # 提取文件名中的数字，如果没有数字则返回0
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def walk_and_sort(directory):
    dir_dict = {}
    for root, dirs, files in os.walk(directory):
        dirs.sort()  # 预先排序目录
        # 按文件名中的数字排序文件
        sorted_files = sorted(files, key=extract_number)
        dir_dict[root] = sorted_files

    sorted_root_dirs = sorted(dir_dict.keys(), key=lambda x: x.lower())

    sorted_paths = []
    for root in sorted_root_dirs:
        for file in dir_dict[root]:
            sorted_paths.append(os.path.join(root, file))

    return {index: path for index, path in enumerate(sorted_paths)}


lines = []
with open('E:\python\lnm\\automation_for_markdown\\test.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        lines.append(json.loads(line))

# Sort the lines based on the first number in each list
sorted_lines = sorted(lines, key=lambda x: x[0])
sorted_files = walk_and_sort("E:\python\lnm\\automation_for_markdown\data_chunks\made with sionna")
for i, instruction in sorted_lines: # 每个chunk的instruction
    if '\n\n' in instruction:
        instruction = instruction.split('\n\n')
    else:
        instruction = instruction.split('\n')
    with open(sorted_files[i], 'r', encoding='utf-8') as f:
        content = f.readlines()
        content = [line for line in content if not line.strip().startswith('<img alt="')]
    with open("E:\python\lnm\\automation_for_markdown\\test.jsonl", 'w', encoding='utf-8'):
        pass  # 打开文件后立即关闭，从而清空文件内容
    for question in instruction: # 每个instruction，循环一个chunk
        question = question.split("INSTRUCTION: ")[1]
        answer = markdown_prompt.answer(question,content)
        jsonl_content = [{"role": "user", "content": answer}]
        with open("E:\python\lnm\\automation_for_markdown\\test.jsonl", 'a', encoding='utf-8') as jsonl_file:
            # Write each JSON object on a new line without indentation
            jsonl_file.write(json.dumps(jsonl_content) + '\n')
    # 一个chunk的写完了, 调用脚本
    with open("E:\python\lnm\\automation_for_markdown\\test_results.jsonl", 'w', encoding='utf-8'):
        pass  # 打开文件后立即关闭，从而清空文件内容
    command = ['python', 'E:\python\lnm\\automation_for_markdown\parallel_request.py', 'E:\python\lnm\\automation_for_markdown\\test.jsonl']
    subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    with open("E:\python\lnm\\automation_for_markdown\\test_results.jsonl", 'r', encoding='utf-8') as f:
        answer_lines = f.readlines()
        answer_dict = {ast.literal_eval(line)[0]: ast.literal_eval(line)[1] for line in answer_lines}
    directory, filename = os.path.split(sorted_files[i])
    new_filename = 'IA_' + filename
    new_path = os.path.join(directory, new_filename)
    # with open(new_path, 'w', encoding='utf-8') as writer:
    #     for p in range(len(instruction)):
    #         writer.write(f"{instruction[p]}\n")
    #         safe_write(writer, f"ANSWER:{answer_dict[p]}\n")
    #         writer.write("\n")
    #         writer.flush()
    for p in range(len(instruction)):
        safe_write(new_path, f"{instruction[p]}\n")
        safe_write(new_path, f"ANSWER:{answer_dict[p]}\n")
        safe_write(new_path, "\n")


