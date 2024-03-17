import matplotlib
matplotlib.use('TkAgg')  # 或其他后端

import matplotlib.pyplot as plt

# 3有些对的，5基本对一般错一半，8基本意思到了
# human = [6.2, 1.03, 1.5, 10, 2,
#          5.5, 5.2, 2, 1.4, 4.3,
#          4.6, 5.2, 0, 6, 1,
#          5.6, 6, 7, 6, 3,
#          7.5, 3, 3, 3.6, 5.8,
#          3, 6.4, 0.5, 3]
human = [6.2, 1.03, 3, 10, 2,
         5.5, 5.2, 2, 1.4, 4.3,
         3, 5.2, 1, 6, 1,
         5.6, 6, 7, 6, 3,
         7.5, 3, 3, 3.6, 5.8,
         6.4, 1, 3]

# GPT4 score
score_list = []
with open('E:\python\lnm\\automation_for_markdown\\test_results.jsonl', 'r') as file:
    lines = file.readlines()

# Convert each line to a list and sort by the first element
sorted_lines = sorted([eval(line.strip()) for line in lines], key=lambda x: x[0])

# Store the sorted lines in a dictionary
sorted_dict = {line[0]: line[1] for line in sorted_lines}

for value in sorted_dict.values():
    value = value.split("\n\n")[0]
    value =value.split("\n")[1]
    score = value.split("Correctness: ")[1]
    score_list.append(int(score))
score_list = score_list[0:30]
score_list.pop(25)
score_list.pop(26)
print(score_list)


plt.figure(figsize=(8, 6))
plt.scatter(human, score_list)
plt.xlabel('Human')
plt.ylabel('Score')
plt.title('Scatter Plot of Human vs Score')
plt.grid(True)

# 添加对角线，以表示完美的一致性
plt.plot([0, 10], [0, 10], 'r--')

plt.show()