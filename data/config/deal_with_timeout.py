import os
import datagen
import time
import litellm.timeout

# Directory containing the files
directory = "E:\python\lnm\\timeout_files"
time_file_path = "/time.txt"

while sum(1 for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))) > 0:
    # List all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                # Call the gen function
                start = time.time()
                datagen.gen(file_path, ("./new/new_chance_"+filename))
                end = time.time()
                print(f"{filename} succeed this time. Information stores in ./new/new_chance_{filename}.")
                ## add the new generate information and delete the temp file
                with open("./new/new_chance_" + filename, 'r',encoding='utf-8') as file2:
                    content_to_append = file2.read()
                with open("./sionna_try/IA_"+filename.split("__")[0], 'a',encoding='utf-8') as file1:
                    file1.write("\n"+content_to_append)
                # change the time, add it
                with open("../../time.txt", 'r') as file:
                    lines_in_time = file.readlines()
                found = False
                for i, line in enumerate(lines_in_time):
                    if ("IA_"+filename.split("__")[0]+":") in line:
                        found = True
                        print(f'for {filename} find line: {line}, modify the time')
                        original_time_in_time = line.split(": ")[1].split(" ")[0]
                        new_time = int(original_time_in_time) + (end - start)
                        line = line.replace(original_time_in_time, str(new_time))
                        line = line.replace(filename.split("__")[1], "")
                        break
                if found:
                    with open("../../time.txt", 'w') as file:
                        file.writelines(lines_in_time)
                    print("time.txt has been updated.")
                else:
                    print(f"The line containing IA_{filename.split('__')[0]}: was not found in the file.")
                    print()
                    print()



                # add the informtion in new to the original and delete it
            except litellm.exceptions.Timeout:
                # Move the file to the timeout_files directory if a timeout occurs
                print(f"Timeout still occurred for {filename}.")
                continue  # Skip the rest of the loop and proceed with the next file
    # Here to delete all the new generated file after a round
    print("Here to delete all the new generated file after a round")
    for filename in os.listdir("../../new"):
        file_path = os.path.join("../../new", filename)
        if os.path.isfile(file_path):
            print(f"Delete {file_path}")
            os.remove(file_path)