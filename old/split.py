import random

input_txt = '/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train_files.txt'
train_txt = '/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt'
val_txt = '/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt'

with open(input_txt, 'r') as f:
    lines = f.read().splitlines()

random.seed(42)  # for reproducibility
random.shuffle(lines)

split_idx = int(0.8 * len(lines))
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open(train_txt, 'w') as f:
    f.write('\n'.join(train_lines) + '\n')

with open(val_txt, 'w') as f:
    f.write('\n'.join(val_lines) + '\n')

print(f"Train set: {len(train_lines)} lines")
print(f"Validation set: {len(val_lines)} lines")