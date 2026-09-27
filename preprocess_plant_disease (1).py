import os
import shutil
import pathlib
import random
from collections import Counter

from PIL import Image
import tensorflow as tf

DATASET_ROOT = r"C:\Users\singh\OneDrive\Documents\Desktop\Plant disease prediction\Dataset"
SPLIT_ROOT = r"C:\Users\singh\OneDrive\Documents\Desktop\Plant disease prediction\dataset_split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)
tf.random.set_seed(SEED)


def inspect_dataset(root):
    root = pathlib.Path(root)
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    counts = {}
    for d in class_dirs:
        n = len([f for f in d.iterdir() if f.is_file()])
        counts[d.name] = n
    print(f"Found {len(class_dirs)} classes, {sum(counts.values())} images total")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:35s} {n:5d} images")
    return counts


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def clean_dataset(root):
    root = pathlib.Path(root)
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            path.unlink()
            removed += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            path.unlink()
            removed += 1
    print(f"Removed {removed} corrupted/invalid files")


def _assert_safe_split_root(root: pathlib.Path, split_root: pathlib.Path):
    root = root.resolve()
    split_root = split_root.resolve()

    if split_root == root:
        raise ValueError(
            "SPLIT_ROOT is identical to DATASET_ROOT. "
            "Choose a different, separate output folder."
        )
    if split_root in root.parents:
        raise ValueError(
            "SPLIT_ROOT is an ancestor of DATASET_ROOT. "
            "Deleting SPLIT_ROOT would delete your raw dataset. "
            "Point SPLIT_ROOT at a separate sibling folder instead."
        )


def split_dataset(root, split_root, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    root = pathlib.Path(root)
    split_root = pathlib.Path(split_root)

    _assert_safe_split_root(root, split_root)

    if split_root.exists():
        shutil.rmtree(split_root)

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir() if f.is_file()]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            out_dir = split_root / split_name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)

    print(f"Split written to '{split_root}/' (train/val/test)")


def build_datasets(split_root, img_size, batch_size):
    split_root = pathlib.Path(split_root)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def prep_train(x, y):
        x = rescale(x)
        x = augment(x, training=True)
        return x, y

    def prep_eval(x, y):
        x = rescale(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    print("=== Step 1: inspecting raw dataset ===")
    inspect_dataset(DATASET_ROOT)

    print("\n=== Step 2: cleaning corrupted images ===")
    clean_dataset(DATASET_ROOT)

    print("\n=== Step 3: creating train/val/test split ===")
    split_dataset(DATASET_ROOT, SPLIT_ROOT, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    print("\n=== Step 4: building tf.data pipelines ===")
    train_ds, val_ds, test_ds, class_names = build_datasets(SPLIT_ROOT, IMG_SIZE, BATCH_SIZE)

    print("\nDone. train_ds / val_ds / test_ds are ready for model.fit().")
    print(f"Number of classes: {len(class_names)}")