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


# def extract_headings_and_intro(lines):
#     pattern = re.compile('[^-#=\*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
#     headings = []
#     intro_content = []
#     intro_started = False
#     intro_end_found = False
#
#     for line in lines:
#         if not intro_started and line.startswith("# "):
#             intro_started = True
#
#         if intro_started and not intro_end_found and line.startswith("## "):
#             intro_end_found = True
#
#         if intro_started and not intro_end_found:
#             intro_content.append(line + '\n')
#         elif line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
#             cleaned_line = pattern.sub('', line.strip())
#             headings.append(cleaned_line + '\n')
#
#     return headings, intro_content
def extract_headings_and_intro(lines):
    # pattern = re.compile('[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
    pattern = re.compile(r'\[source\]|[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
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
            intro_content.append(line + '\n')
        elif line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
            cleaned_line = pattern.sub('', line.strip())
            headings.append(cleaned_line + '\n')

    return headings, intro_content

def find_sections(lines, headings):
    # pattern = re.compile('[^-=#/*\[.\],&:\'()_a-zA-Z0-9"\n ]')
    pattern = re.compile(r'\[source\]|[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        if line.startswith('##'):
            line = pattern.sub('', line)
        if line in headings:  # Check for an exact match, including the # symbols
            current_section = line
            sections[current_section] = []
        if current_section:
            sections[current_section].append(line)
    return sections


def divide_content(sections, intro_content, max_lines=350):
    required_section = intro_content
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
    # pattern = re.compile('[^-#:=*/\[.\],&\'()_a-zA-Z0-9"\n ]')
    pattern = re.compile(r'\[source\]|[^-#=/*\[.\]:,&\'()_a-zA-Z0-9"\n ]')
    headings_in_file = [line for line in content if pattern.sub('', line) in toc_headings]
    # Generate captions based on the hierarchy
    if len(headings_in_file) >=1 and headings_in_file[0].startswith('### '):
        index = toc_headings.index(headings_in_file[0])
        for i in range(index - 1, -1, -1):
            if toc_headings[i].startswith('## '):
                headings_in_file.insert(0, toc_headings[i])
                break
    elif len(headings_in_file) >=1 and headings_in_file[0].startswith('#### '):
        index = toc_headings.index(headings_in_file[0])
        for i in range(index - 1, -1, -1):
            if toc_headings[i].startswith('### '):
                headings_in_file.insert(0, toc_headings[i])
                for j in range(i-1, -1, -1):
                    if toc_headings[j].startswith('## '):
                        headings_in_file.insert(0, toc_headings[j])
                        break
                break
    return headings_in_file


def main():
    time_except_timeout = 0
    timeout_files = []
    whole_content_name = ["api_configuration","api_utility_functions","api_signal","api_ray_tracing","api_5G_NR","api_Multiple-Input Multiple-Output (MIMO)","api_Orthogonal Frequency-Division Multiplexing (OFDM)","api_channel_discrete","api_channel_optical","api_channel_wireless","api_channel","api_mapping","api_forward_error_correction_FEC_utility_functions","api_forward_error_correction_FEC_scrambling","api_forward_error_correction_FEC_interleaving","api_forward_error_correction_FEC_cyclic_redundancy_check","api_forward_error_correction_FEC_turbo_codes","api_forward_error_correction_FEC_convolutional_codes","api_forward_error_correction_FEC_Polar_Codes","api_forward_error_correction_FEC_Low-Density_Parity-Check_LDPC","api_forward_error_correction_FEC_linear_codes","api_forward_error_correction_FEC","api","primer_on_electromagnetics","discover_sionna"]

    for read_content_name in whole_content_name:
        content = read_content("E:\python\lnm\sionna\\" + read_content_name)
        toc_headings, intro_content = extract_headings_and_intro(content)
        sections = find_sections(content, toc_headings)
        divided_contents, intro_content = divide_content(sections, intro_content)
        if read_content_name == 'api' or read_content_name == 'api_forward_error_correction_FEC' or read_content_name == 'api_channel':
            divided_contents = [(content, len(content))]

        for i, (divided_content, _) in enumerate(divided_contents,  start=1):
            file_name = f'data_gen_part_{i}.txt'
            captions = generate_captions_for_file(divided_content, toc_headings)
            file_content = ([intro_content[0]] if intro_content else []) + captions + ["\n"] + divided_content
            write_content(file_name, file_content)
            print(f'Generated {file_name} with {len(file_content)} lines.')

        file_num = len(divided_contents)
        output_file = "IA_" + read_content_name
        for i in range(1, file_num + 1):
        # for i in range(3, 4):
            small_piece_file = f"data_gen_part_{i}.txt"
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
                os.rename(small_piece_file, os.path.join("timeout_files", read_content_name+"__"+small_piece_file))
                continue  # Skip the rest of the loop and proceed with the next file

            print(f"Finished processing {small_piece_file}.")
            os.remove(small_piece_file)  # Remove the file after processing
        end = time.time()
        writer = open('time.txt', 'a', encoding='utf-8')
        if timeout_files:
            writer.write("IA_"+read_content_name+": "+str(time_except_timeout)+" except "+(', '.join(timeout_files))+"\n")
        else:
            writer.write("IA_" + read_content_name + ": " + str(time_except_timeout)+"\n")
        print("finish(may have time out chunks left) of "+read_content_name)
        print()
        timeout_files = []




if __name__ == '__main__':
    main()