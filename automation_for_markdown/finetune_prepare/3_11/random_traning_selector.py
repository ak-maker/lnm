import json

with open("E:\python\lnm\\automation_for_markdown\\finetune_prepare\\train_data.jsonl", 'r', encoding='utf-8') as f:
    training_lines = f.readlines()

total_elements = 1872
samples_to_pick = 100

# 计算抽取的间距
step = total_elements // samples_to_pick

# 使用列表推导式按间距抽取元素
sampled_elements = [i for i in range(0, total_elements, step)]

samples = []
for i in sampled_elements:
    samples.append(training_lines[i])

with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\3_11\sample_test.md', 'w', encoding='utf-8') as f:
    for i in samples:
        data = json.loads(i)
        user_contents = [message['content'] for message in data['messages'] if message['role'] == 'user']
        f.write(str(user_contents[0])+"\n")
