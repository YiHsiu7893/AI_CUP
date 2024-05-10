import random
import os
import shutil

# Define the range
start = 1000000
end = 1004319

# Specify how many random numbers you want to generate
num_numbers = int(4320 * 0.2)  # Change this to the number you need

# Generate random numbers within the specified range
random_numbers = [random.randint(start, end) for _ in range(num_numbers)]

print(random_numbers)

# os.makedirs("train_real", exist_ok=True)
# os.makedirs("train_target", exist_ok=True)
os.makedirs("valid_real", exist_ok=True)
os.makedirs("valid_target", exist_ok=True)

real = "train_real"
target = "train_target"

for i in range(1000000, 1004320, 1):
    source_real = f"{real}/TRA_RI_{i}.png"
    if not os.path.exists(source_real):
        source_real = f"{real}/TRA_RO_{i}.png"
    source_target = f"{target}/TRA_RI_{i}.jpg"
    if not os.path.exists(source_target):
        source_target = f"{target}/TRA_RO_{i}.jpg"
    if i in random_numbers:
        shutil.move(source_real, "valid_real")
        shutil.move(source_target, "valid_target")
