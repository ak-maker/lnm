import os

def convert_to_markdown(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by 'INSTRUCTION:' and 'ANSWER:'
    parts = content.split('INSTRUCTION:')
    markdown_content = ''

    for i, part in enumerate(parts[1:], start=1):
        instruction, answer = part.split('ANSWER:')
        markdown_content += f'### {i}. INSTRUCTION:\n{instruction.strip()}\n\n**ANSWER:**\n\n{answer.strip()}\n\n'

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(markdown_content)

def convert_all_files_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, os.path.splitext(filename)[0] + '.md')
            convert_to_markdown(input_file, output_file)
            print(f"Converted {input_file} to {output_file}")

# Example usage
input_directory = './sionna_intruction_answer'
output_directory = './markdown'
convert_all_files_in_directory(input_directory, output_directory)
