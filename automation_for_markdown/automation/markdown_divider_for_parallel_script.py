from rich import print
import re
import os
import time
import datagen
import litellm.exceptions


# def read_content(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.readlines()
def read_content(file_path): # 去掉开头的<table>直到遇到第一个# {something}, 通常认为是起始标题
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if line.startswith('#'):
                return content[i:]
        return []  # Return an empty list if no line starts with '#'

def write_content(folder_name, file_path, content): # 每次都得改吧，反正分开
    if not os.path.exists("E:\python\lnm\\automation_for_markdown\data_chunks\Tutorials\Ray Tracing\\"+folder_name):
        os.mkdir(f'E:\python\lnm\\automation_for_markdown\data_chunks\Tutorials\Ray Tracing\\{folder_name}')
    with open("E:\python\lnm\\automation_for_markdown\data_chunks\Tutorials\Ray Tracing\\"+folder_name+"\\"+file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)


def extract_headings_from_toc(lines):
    toc = False
    headings = []
    intro_content = []
    name_of_headings = []
    hierarchy_headings = [] #record the hierarchy of captions, 认为起始有两个##，每次多一个tab多一个#
    for line in lines:
        if line.startswith('## Table of Contents') or line.startswith('## Table of contents') or line.startswith('## Table of Content'):
            toc = True
            continue
        if not toc:
            intro_content.append(line)
        else:
            if line.startswith('#') and re.sub(r'^#+ ?', '', line.split('<a class="headerlink" href="')[0]) in name_of_headings: # 出现重复的caption了，即到正文了
                break
            if '- <a class="reference external" href="https://nvlabs.github.io/sionna' in line:
                headings.append(line)
                name_of_headings.append(line.split("\">")[1].split("</a>")[0])
                leading_spaces = len(line) - len(line.lstrip(' '))
                # 将空格数量转换为制表符数量
                tabs = leading_spaces // 2
                hierarchy_headings.append('#'*(tabs+2)+" "+line.split("\">")[1].split("</a>")[0])
    return name_of_headings, hierarchy_headings, intro_content


def find_sections(lines, name_of_headings):
    # pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        if line.startswith('##') and ('<a class="headerlink" href="' in line) and (re.sub(r'^#+ ?', '', line.split('<a class="headerlink" href="')[0]) in name_of_headings):  # 这里默认正文中的caption一定是##开始的
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


def generate_captions_for_file(content, toc_headings, hierarchy_headings):
    # Find all headings in the file content
    # pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    headings_in_file = [line.split('<a class="headerlink" href="')[0] + '\n' for line in content if line.split('<a class="headerlink" href="')[0] in hierarchy_headings]
    # Generate captions based on the hierarchy
    if len(headings_in_file) >=2 and headings_in_file[1].startswith('### '):
        index = hierarchy_headings.index(headings_in_file[1].rstrip('\n'))
        for i in range(index - 1, -1, -1):
            if hierarchy_headings[i].startswith('## '):
                headings_in_file.insert(1, (hierarchy_headings[i]+'\n'))
                break
    elif len(headings_in_file) >=2 and headings_in_file[1].startswith('#### '): ## 这里不对。tutorials_for_experts_MIMO OFDM Transmissions over the CDL Channel Model
        index = hierarchy_headings.index(headings_in_file[1].rstrip('\n'))
        for i in range(index - 1, -1, -1):
            if hierarchy_headings[i].startswith('### '):
                headings_in_file.insert(1, (hierarchy_headings[i]+'\n'))
                for j in range(i-1, -1, -1):
                    if hierarchy_headings[j].startswith('## '):
                        headings_in_file.insert(1, (hierarchy_headings[j]+'\n'))
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
    whole_content_name = ["tutorials_ray_tracing_Introduction to Sionna RT.md","tutorials_ray_tracing_Tutorial on Diffraction.md","tutorials_ray_tracing_Tutorial on Scattering.md"]
    # whole_content_name = ["tutorials_for_experts_End to end Learning with Autoencoders.md"]
    for read_content_name in whole_content_name:
        content = read_content("E:\python\lnm\\automation_for_markdown\markdown\\" + read_content_name)
        content = [line for line in content if not line.startswith('<img alt="')]
        name_of_heading, hierarchy_headings,intro_content = extract_headings_from_toc(content) # hierarchy_headings算出正文的caption有几个’#‘， 这里一个tab两个space
        sections = find_sections(content, name_of_heading)
        divided_contents, intro_content = divide_content(sections, intro_content)

        for i, (divided_content, _) in enumerate(divided_contents, start=1):
            file_name = f'data_gen_part_{i}.md'
            captions = generate_captions_for_file(divided_content, name_of_heading, hierarchy_headings)
            file_content = intro_content + ["# Table of Content\n"] + captions + ["  \n"]+["  \n\n"] + divided_content
            write_content(read_content_name.split(".md")[0],file_name, file_content)
            print(f'Generated {file_name} with {len(file_content)} lines.')




if __name__ == '__main__':
    main()