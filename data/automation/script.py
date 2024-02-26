from rich import print
import re
import os
import time
import datagen
import litellm.exceptions


def read_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_content(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)


def extract_headings_from_toc(lines):
    pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    toc = False
    headings = []
    intro_content = []
    for line in lines:
        if ('# Table of Contents' or '# Table of contents' or '# Table of Content') in line:
            toc = True
            continue
        if not toc:
            intro_content.append(line)
        else:
            if True:
                line = line.split('<')[0] + '\n'
                if pattern.sub('', line) in headings:
                    break
            if line.strip().startswith('#'):
                headings.append(pattern.sub('', line.strip())+'\n')
    return headings, intro_content


def find_sections(lines, headings):
    pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        # if line.startswith('##'):
        #     line = pattern.sub('', line)
        if pattern.sub('',line.split("<")[0]+"\n") in headings:  # Check for an exact match, including the # symbols

            current_section = line
            sections[current_section] = []
        if current_section:
            sections[current_section].append(line)
    return sections


def divide_content(sections, intro_content, max_lines=300):
    # required_section = sections.pop('## GPU Configuration and Imports\n', None)
    required_section = sections.pop('## GPU Configuration and Imports<a class="headerlink" href="https://nvlabs.github.io/sionna/#GPU-Configuration-and-Imports" title="Permalink to this headline">\uf0c1</a>\n', None)
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


def generate_captions_for_file(content, toc_headings):
    # Find all headings in the file content
    pattern = re.compile('[^-#a-zA-Z0-9"\n ]')
    headings_in_file = [line for line in content if pattern.sub('', line) in toc_headings]
    # Generate captions based on the hierarchy
    if len(headings_in_file) >=2 and headings_in_file[1].startswith('### '):
        index = toc_headings.index(headings_in_file[1])
        for i in range(index - 1, -1, -1):
            if toc_headings[i].startswith('## '):
                headings_in_file.insert(1, toc_headings[i])
                break
    elif len(headings_in_file) >=2 and headings_in_file[1].startswith('#### '):
        index = toc_headings.index(headings_in_file[1])
        for i in range(index - 1, -1, -1):
            if toc_headings[i].startswith('### '):
                headings_in_file.insert(1, toc_headings[i])
                for j in range(i-1, -1, -1):
                    if toc_headings[j].startswith('## '):
                        headings_in_file.insert(1, toc_headings[j])
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
    # whole_content_name = ["tutorials_for_experts_MIMO OFDM Transmissions over the CDL Channel Model", "tutorials_for_experts_Neural Receiver for OFDM SIMO Systems", "tutorials_for_experts_Realistic Multiuser MIMO OFDM Simulations","tutorials_for_experts_OFDM MIMO Channel Estimation and Detection","tutorials_for_experts_Introduction to Iterative Detection and Decoding","tutorials_for_experts_End-to-end Learning with Autoencoders","tutorials_for_experts_Weighted Belief Propagation Decoding","tutorials_for_experts_Channel Models from Datasets","tutorials_for_experts_Using the DeepMIMO Dataset with Sionna","tutorials_ray_tracing_Introduction to Sionna RT","tutorials_ray_tracing_Tutorial on Diffraction","tutorials_ray_tracing_Tutorial on Scattering"]
    whole_content_name = ["tutorials_for_experts_OFDM MIMO Channel Estimation and Detection.md"]
    for read_content_name in whole_content_name:
        content = read_content("E:\python\lnm\sionna\\" + read_content_name)
        toc_headings, intro_content = extract_headings_from_toc(content)
        sections = find_sections(content, toc_headings)
        divided_contents, intro_content = divide_content(sections, intro_content)

        for i, (divided_content, _) in enumerate(divided_contents, start=1):
            file_name = f'data_gen_part_{i}.md'
            captions = generate_captions_for_file(divided_content, toc_headings)
            file_content = intro_content + captions + divided_content
            write_content(file_name, file_content)
            print(f'Generated {file_name} with {len(file_content)} lines.')

        file_num = len(divided_contents)
        output_file = "IA_" + read_content_name
    # for i in range(1, file_num+1):
    #     small_piece_file = f"data_gen_part_{i}.txt"
    #     print(f"Processing {small_piece_file}...")
    #     import datagen
    #     datagen.gen(small_piece_file, output_file)
    #     print(f"Finished processing {small_piece_file}.")
    #     os.remove(f"data_gen_part_{i}.txt")
    #     # timeout 记录下来到一个新的folder里里面全是timeout的内容，先执行别的file，最后再执行timeout再把内容加到源文件中
        for i in range(1, file_num + 1):
        # for i in range(3, 4):
            small_piece_file = f"data_gen_part_{i}.md"
            print(f"Processing {small_piece_file}...")

            try:
                # Call the gen function
                start = time.time()
                datagen.gen(small_piece_file, output_file)
                end = time.time()
                time_except_timeout = time_except_timeout + (end - start)
            except litellm.exceptions.Timeout:
                timeout_files.append(small_piece_file)
                # Move the file to the timeout_files directory if a timeout occurs
                print()
                print(f"Timeout occurred for {small_piece_file}. Moving to timeout_files.")
                os.rename(small_piece_file, os.path.join("../../timeout_files", read_content_name + "__" + small_piece_file))
                continue  # Skip the rest of the loop and proceed with the next file

            print(f"Finished processing {small_piece_file}.")
            os.remove(small_piece_file)  # Remove the file after processing
        end = time.time()
        writer = open('../../time_md.txt', 'a', encoding='utf-8')
        if timeout_files:
            writer.write("IA_"+read_content_name+": "+str(time_except_timeout)+" except "+(', '.join(timeout_files))+"\n")
        else:
            writer.write("IA_" + read_content_name + ": " + str(time_except_timeout)+"\n")
        print("finish(may have time out chunks left) of "+read_content_name)
        print()
        timeout_files = []




if __name__ == '__main__':
    main()