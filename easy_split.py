import os
import shutil
import random

# ------------------ CONFIG ------------------

SOURCE_FOLDERS = [
    r"datasets/Montgomery_CXR",
    r"datasets/Shenzhen"
]

EXISTING_TB_DATASET = r"datasets/combined_dataset"

FINAL_DATASET = r"datasets/final_combined_dataset"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_MAP = {
    "0": "Normal",
    "1": "Tuberculosis"
}

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

# --------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collect_images_from_raw(folder):
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith(IMG_EXTENSIONS):

            # Extract label from filename
            label_number = f.split("_")[-1].split(".")[0]
            label_name = LABEL_MAP.get(label_number, "unknown")

            if label_name != "unknown":
                images.append((os.path.join(folder, f), label_name))

    return images


def collect_images_from_split_dataset(base_folder):
    images = []
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(base_folder, split)

        if os.path.exists(split_path):
            for label_folder in os.listdir(split_path):
                label_path = os.path.join(split_path, label_folder)

                if os.path.isdir(label_path):
                    for f in os.listdir(label_path):
                        if f.lower().endswith(IMG_EXTENSIONS):
                            images.append(
                                (os.path.join(label_path, f), label_folder)
                            )
    return images


def collect_all_images():
    all_images = []

    # Raw datasets
    for folder in SOURCE_FOLDERS:
        if os.path.exists(folder):
            all_images += collect_images_from_raw(folder)
        else:
            print(f"Folder not found: {folder}")

    # Existing split dataset
    if os.path.exists(EXISTING_TB_DATASET):
        all_images += collect_images_from_split_dataset(EXISTING_TB_DATASET)

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

            base_name, ext = os.path.splitext(os.path.basename(img_path))
            dest_path = os.path.join(
                dest_dir,
                f"{base_name}_{random.randint(1000,9999)}{ext}"
            )

            shutil.copy(img_path, dest_path)

    print("Final dataset created successfully!")


if __name__ == "__main__":
    print("Collecting images from all datasets...")
    all_images = collect_all_images()
    print(f"Total images collected: {len(all_images)}")

    print("Creating final combined dataset...")
    split_and_copy_images(all_images, FINAL_DATASET)

    print("Done! Dataset ready at:", FINAL_DATASET)
