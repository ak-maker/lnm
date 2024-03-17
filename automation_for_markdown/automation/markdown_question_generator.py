import os
import markdown_prompt
import json

def list_specific_markdown_files(directory, prefix='data_gen'):
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        # 遍历子目录
        for dir in dirs:
            print(f'正在检查目录: {os.path.join(root, dir)}')
            # 构建子目录的完整路径
            sub_dir = os.path.join(root, dir)
            # 遍历子目录中的文件
            for filename in os.listdir(sub_dir):
                if filename.startswith(prefix) and filename.endswith('.md'):
                    print(f'找到Markdown文件: {os.path.join(sub_dir, filename)}')
                    with open(os.path.join(sub_dir, filename), 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    # 排除以 '<img alt="' 开头的行
                    lines = [line for line in lines if not line.strip().startswith('<img alt="')]
                    # store the content to test.jsonl
                    lines = "".join(lines)
                    lines = markdown_prompt.question(lines)
                    print(lines)
                    jsonl_content = [{"role": "user", "content": lines}]
                    # Write to JSONL file
                    with open("E:\python\lnm\\automation_for_markdown\\test.jsonl", 'a', encoding='utf-8') as jsonl_file:
                            # Write each JSON object on a new line without indentation
                            jsonl_file.write(json.dumps(jsonl_content) + '\n')




# 指定要遍历的For Beginners文件夹的路径
directory_path = 'E:\python\lnm\\automation_for_markdown\data_chunks\made with sionna'
list_specific_markdown_files(directory_path)
