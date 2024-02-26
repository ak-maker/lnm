import os

def list_files(directory, strip_suffix=False, strip_prefix=False):
    files = os.listdir(directory)
    if strip_suffix or strip_prefix:
        stripped_files = set()
        for file in files:
            if strip_suffix:
                file = os.path.splitext(file)[0]
            if strip_prefix:
                file = file.removeprefix('IA_')
            stripped_files.add(file)
        return stripped_files
    else:
        return set(files)

def find_unique_files(dir1, dir2):
    files1 = list_files(dir1, strip_suffix=False, strip_prefix=False)
    files2 = list_files(dir2, strip_suffix=True, strip_prefix=True)

    unique_to_dir1 = files1 - files2
    unique_to_dir2 = files2 - files1

    return unique_to_dir1, unique_to_dir2

# Example usage
dir1 = 'E:\python\lnm\sionna'
dir2 = 'E:\python\lnm\markdown'
unique_to_dir1, unique_to_dir2 = find_unique_files(dir1, dir2)

print("Files unique to dir1:", unique_to_dir1)
print("Files unique to dir2:", unique_to_dir2)
