from rich import print
import re
import os
import time
import datagen
import litellm.exceptions



def read_content(file_path): # 去掉开头的<table>直到遇到第一个# {something}, 通常认为是起始标题
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if line.startswith('#'):
                return content[i:]
        return []  # Return an empty list if no line starts with '#'

def write_content(folder_name, file_path, content): # 每次都得改吧，反正分开
    if not os.path.exists("E:\python\lnm\\automation_for_markdown\data_chunks\made with sionna\\"+folder_name):
        os.mkdir(f'E:\python\lnm\\automation_for_markdown\data_chunks\made with sionna\\{folder_name}')
    with open("E:\python\lnm\\automation_for_markdown\data_chunks\made with sionna\\"+folder_name+"\\"+file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)


def extract_headings_and_intro(lines):
    # pattern = re.compile('[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
    # pattern = re.compile(r'\[source\]|[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
    headings = []
    intro_content = []
    intro_started = False
    intro_end_found = False

    for line in lines:
        if not intro_started and line.startswith("# "):
            intro_started = True

        if intro_started and not intro_end_found and line.startswith("## "):
            intro_end_found = True

        if intro_started and not intro_end_found:
            intro_content.append(line)
        elif line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
            # cleaned_line = pattern.sub('', line.strip())
            headings.append(line + '\n')

    return headings, intro_content


def find_sections(lines, name_of_headings):
    # pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        if line.startswith('##') and ('<a class="headerlink" href="' in line) and any(line.rstrip('\n') == heading.rstrip('\n') for heading in name_of_headings):  # 这里默认正文中的caption一定是##开始的
            current_section = line
            sections[current_section] = []
        if current_section:
            sections[current_section].append(line)
    return sections


def divide_content(sections, intro_content, max_lines=300):
    # required_section = sections.pop('## GPU Configuration and Imports\n', None)
    required_section = []
    pattern = re.compile(r'^#+ (.*?(?:import|imports|Import|Imports).*?)<a class="headerlink" href="', re.IGNORECASE)
    if sections and pattern.search(next(iter(sections))):
        # 如果符合，则移除这个元素
        required_section=sections.pop(next(iter(sections)))
    divided_files_content = []
    current_file_content = []
    current_line_count = len(intro_content) if required_section else 0

    for section, content in sections.items():
        if current_line_count + len(content) > max_lines:
            if current_file_content:
                divided_files_content.append((current_file_content, current_line_count))
                current_file_content = []
                current_line_count = 0

        if not current_file_content and required_section:  # Add required section to every new file
            current_file_content.extend(required_section)
            current_line_count += len(required_section)

        current_file_content.extend(content)
        current_line_count += len(content)

    if current_file_content:
        divided_files_content.append((current_file_content, current_line_count))

    return divided_files_content, intro_content


def generate_captions_for_file(content,  hierarchy_headings):
    # Find all headings in the file content
    # pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    hierarchy_headings = [heading.rstrip('\n') for heading in hierarchy_headings]
    headings_in_file = [(line.rstrip('\n')+'\n') for line in content if line.rstrip('\n') in hierarchy_headings]
    if len(headings_in_file) >=2 and headings_in_file[0].startswith('### '):
        index = hierarchy_headings.index(headings_in_file[0].rstrip('\n'))
        for i in range(index - 1, -1, -1):
            if hierarchy_headings[i].startswith('## '):
                headings_in_file.insert(0, (hierarchy_headings[i]+'\n'))
                break
    elif len(headings_in_file) >=2 and headings_in_file[0].startswith('#### '): ## 这里不对。tutorials_for_experts_MIMO OFDM Transmissions over the CDL Channel Model
        index = hierarchy_headings.index(headings_in_file[0].rstrip('\n'))
        for i in range(index - 1, -1, -1):
            if hierarchy_headings[i].startswith('### '):
                headings_in_file.insert(1, (hierarchy_headings[i]+'\n'))
                for j in range(i-1, -1, -1):
                    if hierarchy_headings[j].startswith('## '):
                        headings_in_file.insert(0, (hierarchy_headings[j]+'\n'))
                        break
                break
    # captions = ""
    # for heading in headings_in_file:
    #     if heading not in captions:
    #         captions = captions + heading
    return headings_in_file


def main():
    time_except_timeout = 0
    timeout_files = []
    whole_content_name = ['made with sionna.md']
    # whole_content_name = ["api_forward_error_correction_FEC_interleaving.md"]
    for read_content_name in whole_content_name:
        content = read_content("E:\python\lnm\\automation_for_markdown\markdown\\" + read_content_name)
        content = [line for line in content if not line.startswith('<img alt="')]
        headings, intro_content = extract_headings_and_intro(content) # hierarchy_headings算出正文的caption有几个’#‘， 这里一个tab两个space
        sections = find_sections(content, headings)
        divided_contents, intro_content = divide_content(sections, intro_content)

        for i, (divided_content, _) in enumerate(divided_contents, start=1):
            file_name = f'data_gen_part_{i}.md'
            captions = generate_captions_for_file(divided_content, headings)
            file_content = intro_content + ["# Table of Content\n"] + captions + ["  \n"]+["  \n\n"] + divided_content
            write_content(read_content_name.split(".md")[0],file_name, file_content)
            print(f'Generated {file_name} with {len(file_content)} lines.')




if __name__ == '__main__':
    main()