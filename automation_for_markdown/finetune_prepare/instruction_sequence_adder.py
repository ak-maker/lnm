# 打开并读取文件
with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\test.md', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 给符合条件的Instruction行添加编号
instruction_number = 1
for i in range(len(lines) - 1):
    if lines[i].startswith('Instruction: ') and lines[i + 1].startswith('Answer: '):
        lines[i] = f'**Instruction {instruction_number}:** ' + lines[i][13:]
        instruction_number += 1

# 将修改后的内容写回文件
with open('E:\python\lnm\\automation_for_markdown\\finetune_prepare\\test.md', 'w', encoding='utf-8') as file:
    file.writelines(lines)
