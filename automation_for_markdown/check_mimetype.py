import os

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def search_for_text(directory, text):
    number = 0
    instruction_number = 0
    files = list_all_files(directory)
    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if text in line:
                    print(f"Found in file: {file}, line: {i}")
                    number=number+1
                if 'INSTRUCTION: ' in line:
                    instruction_number = instruction_number + 1
    print("total mimetype number",number)
    print("total instruction number", instruction_number)

# 使用示例
directory = "E:\python\lnm\\automation_for_markdown\data_chunks"
search_for_text(directory, "message='Attempt to decode JSON with unexpected mimetype")





