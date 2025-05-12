import os

data_dir = (
    "/scratch/nadja/cpsc/physionet.org/files/challenge-2020/1.0.2/training/georgia"
)

if os.path.exists(data_dir):
    print(f"✅ Path exists: {data_dir}")
else:
    print(f"❌ Path does NOT exist: {data_dir}")

if os.access(data_dir, os.R_OK):
    print("✅ Read access granted!")
else:
    print("❌ No read access!")

print("Contents of parent directory:", os.listdir(os.path.dirname(data_dir)))
