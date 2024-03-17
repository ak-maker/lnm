import markdown_prompt
import json

instruction = ["Use Snell’s law and vector relationships to calculate the polarization components of the reflected and transmitted wave phasors."]
content_path = 'E:\python\lnm\\automation_for_markdown\data_chunks\primer_on_electromagnetics\data_gen_part_2.md'

with open(content_path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    content = [line for line in content if not line.strip().startswith('<img alt="')]

for instruct in instruction:
    prompt = markdown_prompt.answer(instruct, "".join(content))
    jsonl_content = [{"role": "user", "content": prompt}]
    with open("E:\python\lnm\\automation_for_markdown\\test.jsonl", 'a', encoding='utf-8') as jsonl_file:
        # Write each JSON object on a new line without indentation
        jsonl_file.write(json.dumps(jsonl_content) + '\n')





