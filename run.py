#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd() + os.sep + os.pardir)

import random
import time
import imageio
import numpy as np
import imgaug as ia
from imgaug.augmentables.batches import UnnormalizedBatch
from imgaug import augmenters as iaa

random.seed(42)
ia.seed(42)

file_paths = {
    "source": "src/main/data/source",
    "reshaped": "src/main/data/reshaped",
    "test_augmented": "src/main/data/test_augmented",
    "train_augmented": "src/main/data/train_augmented",
    "validation_augmented": "src/main/data/validation_augmented"
}

TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.10
VALIDATION_SPLIT = 0.10
NB_BATCHES = 10
BATCH_SIZE = 200
NB_TRAIN_SAMPLES = 20000


def get_train_test_validation_file_lists(files_list):
    qty_valid_source_files = len(files_list)

    qty_train_records = int(qty_valid_source_files * TRAIN_SPLIT)
    qty_test_records = int(qty_valid_source_files * TEST_SPLIT)
    qty_validation_records = int(qty_valid_source_files * VALIDATION_SPLIT)

    list_train_files = files_list[:qty_train_records]
    del files_list[:qty_train_records]

    list_test_files = files_list[:qty_test_records]
    del files_list[:qty_test_records]

    list_validation_files = files_list[:qty_validation_records]
    del files_list[:qty_validation_records]

    return list_train_files, list_test_files, list_validation_files


def get_images_batch_stack(image, batch_size=BATCH_SIZE):
    return [np.copy(image) for _ in range(batch_size)]


def get_image_batches(images, nb_batches=NB_BATCHES):
    return [UnnormalizedBatch(images=images) for _ in range(nb_batches)]


def reshape_images(image_name,
                   source_path=file_paths["source"],
                   reshaped_path=file_paths["reshaped"],
                   ):
    image_source_path = source_path + '/' + image_name
    image_reshaped_path = reshaped_path + '/' + image_name

    resize_aug \
        = iaa.Sequential([
        iaa.Resize(256, interpolation=["linear"])
    ])

    raw_image = imageio.imread(image_source_path)

    resize_aug_image \
        = resize_aug(image=raw_image)

    imageio.imwrite(image_reshaped_path, resize_aug_image)


def reshape_save_source_images(file_path=file_paths["source"]):
    list_source_files \
        = os.listdir(file_path)

    list_valid_source_files \
        = [s for s in list_source_files if "jpg" in s]

    for image in list_valid_source_files:
        reshape_images(image_name=image)


def train_test_validation_split_sets(file_path=file_paths["reshaped"]):
    list_reshaped_files \
        = os.listdir(file_path)

    random.shuffle(list_reshaped_files)

    qty_reshaped_files = len(list_reshaped_files)

    if qty_reshaped_files < 10:
        raise Exception(f'Less than 10 files in the reshaped folder. Number of files: {qty_reshaped_files}')

    list_train_files, list_test_files, list_validation_files \
        = get_train_test_validation_file_lists(list_reshaped_files)

    return list_train_files, list_test_files, list_validation_files


random_mixed_effects_aug \
    = iaa.Sequential(
    [iaa.Affine(rotate=(-60, 60)),
     iaa.AdditiveGaussianNoise(scale=(10, 60)),
     iaa.Crop(percent=(0, 0.2))
     ])


def generate_batch_augmentation(image_name,
                                augmentation_effect,
                                reshaped_path=file_paths["reshaped"]):
    image_reshaped_path = reshaped_path + '/' + image_name
    raw_image = imageio.imread(image_reshaped_path)
    stacked_images = get_images_batch_stack(raw_image)
    image_batches = get_image_batches(stacked_images)
    batches_aug \
        = list(augmentation_effect.augment_batches(image_batches,
                                                   background=True))
    return batches_aug


def generate_mix_effects_aug(
        image_name_with_extention,
        effect_name,
        effect_object,
        split_set,
        destination_path,
        reshaped_path=file_paths["reshaped"],
        nb_batchs=NB_BATCHES,
        batch_sizes=BATCH_SIZE):
    image_name \
        = image_name_with_extention.split('.')[0]

    image_aug_batch \
        = generate_batch_augmentation(
        image_name_with_extention,
        effect_object
    )

    for nb_batch in range(0, nb_batchs):
        for batch_size in range(0, batch_sizes):
            image_name_destination_path_with_effect \
                = destination_path + f'/{image_name}_{effect_name}_nb_batch{nb_batch}_batch_size_{batch_size}_{split_set}.jpg'

            imageio.imwrite(
                image_name_destination_path_with_effect,
                image_aug_batch[nb_batch].images_aug[batch_size]
            )


def generate_files_mixed_effects(list_set_files, effect_aug,
                                 effect_name, effect_object,
                                 split_set, destination_path):
    start_time = time.time()

    for image in list_set_files:
        effect_aug(
            image_name_with_extention=image,
            effect_name=effect_name,
            effect_object=effect_object,
            split_set=split_set,
            destination_path=destination_path
        )

    time_elapsed = time.time() - start_time
    print(f'Time elapsed - {split_set} set: {time.strftime("%H:%M:%S", time.gmtime(time_elapsed))}')


def main():
    print("Start Sirius Augmentaton wrapper...")

    print("Reshape source images for 256x256 definition")
    reshape_save_source_images()

    print("Split reshaped images in Train, Test and Validation sets")
    list_train_files, list_test_files, list_validation_files \
        = train_test_validation_split_sets()

    print("Generate mixed effects for the Training Set")
    generate_files_mixed_effects(
        list_set_files=list_train_files,
        effect_aug=generate_mix_effects_aug,
        effect_name='random_mixed_effects_aug',
        effect_object=random_mixed_effects_aug,
        split_set='train',
        destination_path=file_paths["train_augmented"]
    )

    print("Generate mixed effects for the Test Set")
    generate_files_mixed_effects(
        list_set_files=list_test_files,
        effect_aug=generate_mix_effects_aug,
        effect_name='random_mixed_effects_aug',
        effect_object=random_mixed_effects_aug,
        split_set='test',
        destination_path=file_paths["test_augmented"]
    )

    print("Generate mixed effects for the Validation Set")
    generate_files_mixed_effects(
        list_set_files=list_validation_files,
        effect_aug=generate_mix_effects_aug,
        effect_name='random_mixed_effects_aug',
        effect_object=random_mixed_effects_aug,
        split_set='validation',
        destination_path=file_paths["validation_augmented"]
    )

    print("Sirius Augmentaton wrapper finished")


if __name__ == "__main__":
    main()
