import json

# Read the second line of test.jsonl
with open('E:\python\lnm\\automation_for_markdown\\test.jsonl', 'r',encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i == 1:  # Second line has index 1
            second_line = json.loads(line)
            for item in second_line:
                if 'content' in item:
                    content = item['content']
                    break
            break

# Write the content to a new file
with open('/automation_for_markdown/test.md', 'w', encoding='utf-8') as output_file:
    output_file.write(content)


