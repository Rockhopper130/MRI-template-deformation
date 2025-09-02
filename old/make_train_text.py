main_dir = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/"
input_file = "train_files.txt"
output_file = "train_files.txt"
prefix = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/"
suffix = "/seg4_onehot.npy"

with open(main_dir + input_file, 'r') as f:
    lines = f.read().splitlines()

updated_lines = [
    f"{prefix}{line}{suffix}" for line in lines if line.strip() != "OASIS_OAS1_0406_MR1"
]

print(updated_lines)

with open(main_dir + output_file, 'w') as f:
    f.write("\n".join(updated_lines) + "\n")
