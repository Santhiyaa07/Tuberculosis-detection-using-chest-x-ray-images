import os
import shutil
import random

# ------------------ CONFIG ------------------
# Source folders for Montgomery/Shenzhen images
SOURCE_FOLDERS = [
    r"datasets/Montgomery_CXR",  # Make sure this folder exists
    r"datasets/Shenzhen"          # Make sure this folder exists
]

# Existing combined dataset
COMBINED_DATASET = r"datasets/combined_dataset"

# Split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Mapping from filename start 0/1 → class names (match your folder names)
LABEL_MAP = {
    "0": "Normal",
    "1": "Tuberculosis"
}
# --------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_and_split_images(source_folder, combined_folder):
    # List all images
    images = [f for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)

    # Split counts
    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)
    test_count = total - train_count - valid_count

    splits = {
        "train": images[:train_count],
        "valid": images[train_count:train_count + valid_count],
        "test": images[train_count + valid_count:]
    }

    for split_name, split_images in splits.items():
        for img in split_images:
            label_number = img.split(".")[0]  # assume filename starts with 0 or 1
            label_name = LABEL_MAP.get(label_number, "unknown")
            if label_name == "unknown":
                continue

            dest_dir = os.path.join(combined_folder, split_name, label_name)
            ensure_dir(dest_dir)

            src_path = os.path.join(source_folder, img)
            base_name, ext = os.path.splitext(img)
            dest_path = os.path.join(dest_dir, f"{base_name}_{random.randint(1000,9999)}{ext}")
            shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    for folder in SOURCE_FOLDERS:
        if os.path.exists(folder):
            print(f"Processing {folder} ...")
            copy_and_split_images(folder, COMBINED_DATASET)
        else:
            print(f"Folder not found: {folder}")

    print("All images processed and added to combined dataset!")
