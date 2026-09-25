import os

def generate_sh_files(directory, num_files):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(1, num_files + 1):
        file_name = f"data{i}.sh"
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --cpus-per-task=1\n")
            file.write("#SBATCH --ntasks-per-node=1\n")
            file.write("#SBATCH --mem-per-cpu=10GB\n")
            file.write("#SBATCH --time=48:00:00\n")
            file.write("#SBATCH --output="+str(i)+".out\n\n")
            file.write("module --ignore-cache load \"miniconda-nobashrc\"\n")
            file.write("eval \"$(conda shell.bash hook)\"\n")
            file.write("conda activate multinest\n\n")
            start = (i - 1) * 6000
            end = i * 6000
            file.write(f"python call_class.py --start {start} --end {end}\n")

if __name__ == "__main__":
    output_directory = "/Users/mengxiwu/Desktop/class_public"
    number_of_files = 10  # Change this to the number of files you want to generate
    generate_sh_files(output_directory, number_of_files)