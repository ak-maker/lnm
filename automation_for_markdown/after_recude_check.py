# 因为reduce完GPT可能不完全按照instrction-answer的形式写，在此检查
import json

# 打开文件并逐行读取
with open('./test_results.jsonl', 'r') as file:
    for i, line in enumerate(file, start=1):
        # 将每一行解析为JSON对象
        data = json.loads(line)
        # 获取列表的第二个元素，并按照'\n'或者'\n\n'分割
        instructions = data[1].split('\n\n') if '\n\n' in data[1] else data[1].split('\n')
        # 检查最后一个分割元素的前几个字符是否为'INSTRUCTION: '
        if not instructions[-1].startswith('INSTRUCTION: '):
            # 如果不是，则打印行数
            print(f'Line {i} does not start with "INSTRUCTION: "')
