import os
import random
import shutil
import tensorflow as tf

INPUT_DATASET = r"C:\Users\singh\OneDrive\Documents\Desktop\Plant disease prediction\Dataset"
OUTPUT_DATASET = r"C:\Users\singh\OneDrive\Documents\Desktop\Plant disease prediction\processed_dataset"

IMAGE_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
AUGMENT_PER_IMAGE = 2

random.seed(42)


augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1)
])

for disease in os.listdir(INPUT_DATASET):

    disease_path = os.path.join(INPUT_DATASET, disease)

    if not os.path.isdir(disease_path):
        continue

    images = []

    for file in os.listdir(disease_path):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            images.append(file)


    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)

    validation_end = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:validation_end
    ]

    test_images = images[
        validation_end:
    ]

    train_folder = os.path.join(
        OUTPUT_DATASET,
        "train",
        disease
    )

    validation_folder = os.path.join(
        OUTPUT_DATASET,
        "validation",
        disease
    )

    test_folder = os.path.join(
        OUTPUT_DATASET,
        "test",
        disease
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)


    for file in train_images:

        path = os.path.join(
            disease_path,
            file
        )

        image = tf.keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE
        )

        image = tf.keras.utils.img_to_array(image)
        original = tf.keras.utils.array_to_img(image)

        original.save(
            os.path.join(
                train_folder,
                file
            )
        )

        for i in range(AUGMENT_PER_IMAGE):

            augmented = augmentation(
                tf.expand_dims(image, 0),
                training=True
            )[0]

            augmented = tf.clip_by_value(
                augmented,
                0,
                255
            )

            augmented = tf.keras.utils.array_to_img(
                augmented
            )

            new_name = (
                f"aug_{i}_{file}"
            )

            augmented.save(
                os.path.join(
                    train_folder,
                    new_name
                )
            )

    for file in validation_images:

        path = os.path.join(
            disease_path,
            file
        )

        image = tf.keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE
        )

        image.save(
            os.path.join(
                validation_folder,
                file
            )
        )

    for file in test_images:

        path = os.path.join(
            disease_path,
            file
        )

        image = tf.keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE
        )

        image.save(
            os.path.join(
                test_folder,
                file
            )
        )

    print(
        disease,
        " | Train:",
        len(train_images),
        "| Validation:",
        len(validation_images),
        "| Test:",
        len(test_images)
    )


print("\n================================")
print("PREPROCESSING COMPLETED")
print("================================")
print("Dataset saved at:", OUTPUT_DATASET)
