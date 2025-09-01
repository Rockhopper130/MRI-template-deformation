input_file = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
output_file = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/train.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    parts = line.strip().split("/")
    for part in parts:
        if part.startswith("OASIS_OAS1_0"):
            cleaned.append(part)
            break

with open(output_file, "w") as f:
    f.write("\n".join(cleaned))

input_file = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt"
output_file = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/val.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    parts = line.strip().split("/")
    for part in parts:
        if part.startswith("OASIS_OAS1_0"):
            cleaned.append(part)
            break

with open(output_file, "w") as f:
    f.write("\n".join(cleaned))