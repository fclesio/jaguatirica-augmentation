#!/usr/bin/env python
# coding: utf-8

# Load image processing Libraries
import imageio
import cv2 as cv
import imgaug as ia
import numpy as np
import logging
import os
import time
import uuid
import shutil
from imgaug import augmenters as iaa
import multiprocessing as mp

# Cleanup log file if exists
try:
    open("log-jaguatirica.log", "w").close()
except OSError:
    pass

# Logging
logging.basicConfig(filename=f'log-jaguatirica.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Definition of all constants and paths
ia.seed(42)
np.random.seed(seed=42)
n_images = 15

# Cleanup folders
start_time = time.time()
logger.info(f'DATA PREPARATION - Cleanup folders: {time.strftime("%H:%M:%S" , time.gmtime(start_time))}')

# All image paths
ROOT_DIR = os.getcwd()
SOURCE_DIR = '/source/'
DESTINATION_DIR = '/destination/'
ROTATED_DIR = '/rotated/'

# Create working folders
working_folders = ['/destination', '/rotated']
for folder in working_folders:
    shutil.rmtree(ROOT_DIR + folder, ignore_errors=True)
    os.mkdir(ROOT_DIR + folder)


elapsed_time = time.time() - start_time
logger.info(f'DATA PREPARATION - Cleanup folders - Elapsed time: {time.strftime("%H:%M:%S",time.gmtime(elapsed_time))}')

logger.info(f'DATA PREPARATION - Source folder: {ROOT_DIR + SOURCE_DIR}')
logger.info(f'DATA PREPARATION - Destination folder: {ROOT_DIR + DESTINATION_DIR}')
logger.info(f'DATA PREPARATION - Rotated folder: {ROOT_DIR + ROTATED_DIR}')

# List all files
array_files = os.listdir(ROOT_DIR + SOURCE_DIR)

# Filtering out not .jpg files from the array
array_images = [s for s in array_files if "jpg" in s]

if not isinstance(array_images, list):
    raise ValueError(f'DATA PREPARATION - Files not in a list. {time.strftime("%H:%M:%S",time.gmtime(time.time()))}')


def get_rotation(image):
    """
    Augmentation performs a initial rotation.

    Initial rotation where a human can read without effort.
    As we're using 45 degrees left and right, I'll put 120
    as the maximum combination of all rotations

    Parameters
    -----------
    image: string
        path for a .jpg file

    """

    rotation_degrees = 45
    rotation_combinations = 90
    img_source = ROOT_DIR + SOURCE_DIR + image

    logger.info(f'AUGMENTATION - Rotation - Starting {img_source} image')

    image = imageio.imread(img_source)
    seq = iaa.Sequential([iaa.Affine(rotate=(-rotation_degrees, rotation_degrees)),
                          ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(rotation_combinations)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + ROTATED_DIR + f'image_rotated_{str(uuid.uuid4().hex)}.jpg', images_aug[i])

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_rotated_{str(uuid.uuid4().hex)}.jpg', images_aug[i])

    logger.info(f'AUGMENTATION - Rotation - {img_source} finished')


def get_gaussian_noise(image):
    """
    Input a gaussian noise in the image

    Add gaussian noise to an image, sampled once per
    pixel from a normal distribution N(0, s),
    where s is sampled per image and varies
    between 0 and 0.05*255

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    gaussian_noise_lower = 30
    gaussian_noise_upper = 90
    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(gaussian_noise_lower, gaussian_noise_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_gaussian_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_crop_images(image):
    """
    Crop the image

    Crop images from each side by 0
    to 50% (randomly chosen)

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    crop_lower = 0
    crop_upper = 0.3
    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.Crop(percent=(crop_lower, crop_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_crop_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_flip_image(image):
    """
    Flip/mirror input images horizontally.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """

    # Horizontally flip n% of the images
    horizontal_flip_degrees = 0.2

    # Vertically flip n% of the images
    vertical_flip_degrees = 0.2

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq_flip_horiz = iaa.Sequential([iaa.Fliplr(horizontal_flip_degrees),
                                      ], random_order=True)

    seq_flip_verti = iaa.Sequential([iaa.Flipud(vertical_flip_degrees),
                                      ], random_order=True)

    images_aug_horiz = [seq_flip_horiz.augment_image(image) for _ in range(n_images)]
    images_aug_verti = [seq_flip_verti.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug_horiz) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_flip_horiz_{str(uuid.uuid4().hex)}.jpg',
                        images_aug_horiz[i])

    for i in range(0, len(images_aug_verti) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_flip_verti_{str(uuid.uuid4().hex)}.jpg',
                        images_aug_verti[i])


def get_blur_image(image):
    """
    Blur the image using gaussian kernels.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """

    blur_sigma_lower = 0
    blur_sigma_upper = 2.5

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.GaussianBlur(sigma=(blur_sigma_lower, blur_sigma_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_blur_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_normalization_image(image):
    """
    Normalize the image making changes in contrast.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    normalization_lower = 0.75
    normalization_upper = 9.5

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.ContrastNormalization((normalization_lower, normalization_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images) ]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_normalization_{str(uuid.uuid4().hex)}.jpg',
                        images_aug[i])


def get_sharpen_image(image):
    """
    Augmenter that sharpens images and overlays
    the result with the original image

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    sharpen_alpha_lower = 0
    sharpen_alpha_upper = 1
    sharpen_lightness_lower = 0.75
    sharpen_lightness_upper = 1.5

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.Sharpen(alpha=(sharpen_alpha_lower, sharpen_alpha_upper),
                                       lightness=(sharpen_lightness_lower, sharpen_lightness_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_sharpen_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_channel_image(image):
    """
    Add a value to all pixels in an image inside a range.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    add_lower = -10
    add_upper = 10

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.Add((add_lower , add_upper), per_channel=0.5),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_channel_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_gray_image(image):
    """
    Transform the image in gray scale

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    grayscale_alpha_lower = 0.0
    grayscale_alpha_upper = 1.0

    image = cv.imread(ROOT_DIR + SOURCE_DIR + image, 1)
    seq = iaa.Sequential([iaa.Grayscale(alpha=(grayscale_alpha_lower, grayscale_alpha_upper))
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_grey_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_water_image(image):
    """
    Include the water effect in the image. Transform images by
    moving pixels locally around using displacement fields.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    transformation_alpha = 50
    transformation_sigma = 9

    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=transformation_alpha, sigma=transformation_sigma),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_water_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def get_negative_image(image):
    """
    Include the "negative film" effect in the image.
    Augmenter that increases/decreases hue and
    saturation by random values.

    Parameters
    -----------
    image: string
        path for a .jpg file

    """
    saturation_lower = -40
    saturation_upper = 40

    image = cv.imread(ROOT_DIR + SOURCE_DIR + image, 1)
    seq = iaa.Sequential([iaa.AddToHueAndSaturation((saturation_lower, saturation_upper)),
                           ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug) - 1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR + f'image_negative_{str(uuid.uuid4().hex)}.jpg', images_aug[i])


def main_initial_rotation():
    """
    Rotate all images

    Main wrapper for initial rotation for all images

    """
    pool = mp.Pool(mp.cpu_count())
    pool.map(get_rotation, array_images)


# Time tracking
start_time = time.time()
logger.info(f'AUGMENTATION - Rotation - Start: {time.strftime("%H:%M:%S" , time.gmtime(start_time))}')

# Call the main wrapper with multiprocessing
main_initial_rotation()

elapsed_time = time.time() - start_time
logger.info(f'AUGMENTATION - Rotation - Finished: {time.strftime("%H:%M:%S" , time.gmtime(elapsed_time))}')


def main_apply_effects():
    """
    Apply functions in rotated images

    Main wrapper for effects functions over rotated images

    """
    pool = mp.Pool(mp.cpu_count())
    pool.map(get_gaussian_noise, array_images)
    pool.map(get_crop_images, array_images)
    pool.map(get_flip_image, array_images)
    pool.map(get_blur_image, array_images)
    pool.map(get_normalization_image, array_images)
    pool.map(get_sharpen_image, array_images)
    pool.map(get_channel_image, array_images)
    pool.map(get_gray_image, array_images)
    pool.map(get_water_image, array_images)
    pool.map(get_negative_image, array_images)


# Time tracking
start_time = time.time()
logger.info(f'AUGMENTATION - Start apply effects: {time.strftime("%H:%M:%S" , time.gmtime(start_time))}')

# Call the main wrapper with multiprocessing
main_apply_effects()

elapsed_time = time.time() - start_time
logger.info(f'AUGMENTATION - Apply effects - Elapsed time: {time.strftime("%H:%M:%S" , time.gmtime(elapsed_time))}')
logger.info(f'AUGMENTATION - End apply effects - Timestamp: {time.strftime("%H:%M:%S" , time.gmtime(time.time()))}')
