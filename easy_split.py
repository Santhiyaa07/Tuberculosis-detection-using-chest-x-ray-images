import os
import shutil
import random

# ------------------ CONFIG ------------------
# Source folders with 0/1 labels
SOURCE_FOLDERS = [
    r"datasets/Montgomery_CXR",
    r"datasets/Shenzhen"
]


# Existing pre-split TB dataset folders
EXISTING_TB_DATASET = r"datasets/combined_dataset"

# Final combined dataset folder
FINAL_DATASET = r"datasets/final_combined_dataset"

# Split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Mapping 0/1 → class names
LABEL_MAP = {
    "0": "Normal",
    "1": "Tuberculosis"
}

# Supported image extensions
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")
# --------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_images_from_folder(folder, label_map=None):
    """
    Collects images from a folder.
    If label_map is provided, converts 0/1 to normal/tuberculosis.
    Returns list of tuples: (image_path, label_name)
    """
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith(IMG_EXTENSIONS):
            if label_map:
                # filename assumed to start with 0 or 1
                label_number = f.split(".")[0]
                label_name = label_map.get(label_number, "unknown")
                if label_name == "unknown":
                    continue
            else:
                # if already labeled folder
                label_name = os.path.basename(folder)
            images.append((os.path.join(folder, f), label_name))
    return images

def collect_all_images():
    all_images = []

    # 1. Collect images from Montgomery and Shenzhen
    for folder in SOURCE_FOLDERS:
        if os.path.exists(folder):
            all_images += collect_images_from_folder(folder, LABEL_MAP)
        else:
            print(f"Folder not found: {folder}")

    # 2. Collect images from existing combined TB dataset
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(EXISTING_TB_DATASET, split)
        if os.path.exists(split_path):
            for label_folder in os.listdir(split_path):
                label_path = os.path.join(split_path, label_folder)
                if os.path.isdir(label_path):
                    all_images += collect_images_from_folder(label_path)
    return all_images

def split_and_copy_images(all_images, final_folder):
    random.shuffle(all_images)
    total = len(all_images)
    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)
    test_count = total - train_count - valid_count

    splits = {
        "train": all_images[:train_count],
        "valid": all_images[train_count:train_count + valid_count],
        "test": all_images[train_count + valid_count:]
    }

    for split_name, images in splits.items():
        for img_path, label_name in images:
            dest_dir = os.path.join(final_folder, split_name, label_name)
            ensure_dir(dest_dir)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(dest_dir, f"{base_name}_{random.randint(1000,9999)}{ext}")
            shutil.copy(img_path, dest_path)

if __name__ == "__main__":
    print("Collecting all images from all sources...")
    all_images = collect_all_images()
    print(f"Total images collected: {len(all_images)}")

    print("Splitting and copying images into final combined dataset...")
    split_and_copy_images(all_images, FINAL_DATASET)
    print("Done! Final combined dataset created at:", FINAL_DATASET)
