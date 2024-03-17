import os

def merge_ia_md_files(input_directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.startswith("IA_") and file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read() + '\n\n')

input_directory = r"E:\python\lnm\automation_for_markdown\data_chunks"
output_file = r"E:\python\lnm\automation_for_markdown\finetune_prepare\IA_total.md"
merge_ia_md_files(input_directory, output_file)
