import re


def read_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_content(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)


def extract_headings_from_toc(lines):
    toc = False
    headings = set()
    intro_content = []
    for line in lines:
        if '# Table of Contents' in line:
            toc = True
            continue
        if not toc:
            intro_content.append(line)
        else:
            if line.startswith('# ') and not '# Table of Contents' in line:
                break
            if line.strip().startswith('#'):
                headings.add(line.strip())
    return headings, intro_content


def find_sections(lines, headings):
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        if line.strip() in headings:  # Check for an exact match, including the # symbols
            current_section = line.strip()
            sections[current_section] = []
        if current_section:
            sections[current_section].append(line)
    return sections


def divide_content(sections, intro_content, max_lines=450):
    required_section = sections.pop('## GPU Configuration and Imports', None)
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
    headings_in_file = [line for line in content if line.strip() in toc_headings]
    # Generate captions based on the hierarchy
    captions = []
    for heading in headings_in_file:
        if heading not in captions:
            captions.append(heading)
    return captions


def main():
    content = read_content('context.txt')
    toc_headings, intro_content = extract_headings_from_toc(content)
    sections = find_sections(content, toc_headings)
    divided_contents, intro_content = divide_content(sections, intro_content)

    for i, (divided_content, _) in enumerate(divided_contents, start=1):
        file_name = f'data_gen_part_{i}.txt'
        captions = generate_captions_for_file(divided_content, toc_headings)
        file_content = intro_content + captions + divided_content
        write_content(file_name, file_content)
        print(f'Generated {file_name} with {len(file_content)} lines.')


if __name__ == '__main__':
    main()