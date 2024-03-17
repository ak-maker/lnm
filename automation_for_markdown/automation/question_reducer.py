import json
import markdown_prompt as markdown_prompt
# get the question from the content by hand, stored in test.jsonl(need to run results_to_test_convertor.py)
# Now reduce the question
with open('E:\python\lnm\\automation_for_markdown\\test.jsonl', 'r', encoding='utf-8') as file:
    lines = [json.loads(line)[0] for line in file]


for line in lines:
    line['content'] = markdown_prompt.reduce(line['content'])

# Write the modified content back to test.jsonl
with open('E:\python\lnm\\automation_for_markdown\\test.jsonl', 'w') as file:
    for line in lines:
        file.write('['+json.dumps(line)+']' + '\n')