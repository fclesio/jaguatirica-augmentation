#!/usr/bin/env python
# coding: utf-8

# Load image processing Libraries
import cv2 as cv
import glob
import imageio
import imgaug as ia
import numpy as np
import os
import random
import time
import uuid
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

ia.seed(42)
np.random.seed(seed=42)


# ### Cleanup folders
start_time = time.time()

ROOT_DIR = os.getcwd()

cleanup_folders = ['/DESTINATION/*', '/ROTATED/*']

for folder in cleanup_folders:
    files = glob.glob(ROOT_DIR + folder)
    for f in files:
        os.remove(f)
    
SOURCE_DIR = '/SOURCE/'
DESTINATION_DIR = '/DESTINATION/'

# The main transformation will be the rotation, i.e.
# every image will be all combinations of rotation
# from -60 until + 60 degrees. I'll implemement
# these fixed values because empirically I noticed
# that these degrees can be readble for a human being
# in front some computer without any problem or effort
ROTATED_DIR = '/ROTATED/'

elapsed_time = time.time() - start_time
print(f'Elapsed Time - Cleanup folders: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


# ### Augmentation - Initial Rotation
start_time = time.time()

# List all files
source_files = os.listdir(ROOT_DIR + SOURCE_DIR)

# Filtering out not .jpg files from the array
source_images = [s for s in source_files if "jpg" in s]


def get_rotation(image, image_name):
    # Initial rotation where a human can read without effort. 
    # As we're using 60 degrees left and right, I'll put 120
    # as the maximum combination of all rotations
    rotation_degrees=60
    rotation_combinations=120
    
    seq = iaa.Sequential([iaa.Affine(rotate=(-rotation_degrees, rotation_degrees)),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(rotation_combinations)]

    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + ROTATED_DIR                         
                        + f'image_rotated_{i}_{str(image_name)}', images_aug[i])
        
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_rotated_{i}_{str(image_name)}', images_aug[i])        
    
        
# Rotate all images
for image in source_images:
    image_name=image
    image = imageio.imread(ROOT_DIR + SOURCE_DIR + image)
    get_rotation(image=image, image_name=image_name)
    
elapsed_time = time.time() - start_time
print(f'Elapsed Time - Rotation Files: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')    


# ### Augmentation - General functions
# 
# - `get_gaussian_noise(image, n_images)`
# - `get_crop_images(image, n_images)`
# - `get_flip_image(image, n_images)`
# - `get_blur_image(image, n_images)`
# - `get_normalization_image(image, n_images)`
# - `get_sharpen_image(image, n_images)`
# - `get_channel_image(image, n_images)`
# - `get_gray_image(image, n_images)`
# - `get_water_image(image, n_images)`
# - `get_negative_image(image, n_images)`
# - `get_random_agumentation()`


def get_gaussian_noise(image, n_images):
    '''Add gaussian noise to an image, sampled once per 
    pixel from a normal distribution N(0, s), 
    where s is sampled per image and varies 
    between 0 and 0.05*255'''
    
    gaussian_noise_lower = 30
    gaussian_noise_upper = 90

    seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(gaussian_noise_lower,gaussian_noise_upper)),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_gaussian_{str(uuid.uuid4().hex)}.jpg',images_aug[i])
        
        
def get_crop_images(image, n_images):
    ''' Crop images from each side by 0 
    to 50% (randomly chosen)'''
    crop_lower = 0
    crop_upper = 0.3

    seq = iaa.Sequential([iaa.Crop(percent=(crop_lower, crop_upper)), 
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_crop_{str(uuid.uuid4().hex)}.jpg', images_aug[i]) 
        
        
def get_flip_image(image, n_images):
    # Horizontally flip n% of the images
    horizontal_flip_degrees = 0.2
    
    # Vertically flip n% of the images
    vertical_flip_degrees = 0.2

    seq_flip_horiz = iaa.Sequential([iaa.Fliplr(horizontal_flip_degrees), 
                         ], random_order=True)
    
    seq_flip_verti = iaa.Sequential([iaa.Flipud(vertical_flip_degrees), 
                     ], random_order=True)

    images_aug_horiz = [seq_flip_horiz.augment_image(image) for _ in range(n_images)]
    images_aug_verti = [seq_flip_verti.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug_horiz)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_flip_horiz_{str(uuid.uuid4().hex)}.jpg',                        
                        images_aug_horiz[i])
        
    for i in range(0, len(images_aug_verti)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_flip_verti_{str(uuid.uuid4().hex)}.jpg',                        
                        images_aug_verti[i])   
        
        
def get_blur_image(image, n_images):
    blur_sigma_lower = 0
    blur_sigma_upper = 2.5

    seq = iaa.Sequential([iaa.GaussianBlur(sigma=(blur_sigma_lower, blur_sigma_upper)), 
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_blur_{str(uuid.uuid4().hex)}.jpg', images_aug[i])   
        
        
def get_normalization_image(image, n_images):
    normalization_lower = 0.75
    normalization_upper = 9.5

    seq = iaa.Sequential([iaa.ContrastNormalization((normalization_lower, normalization_upper)),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_normalization_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        
        
def get_sharpen_image(image, n_images):
    sharpen_alpha_lower = 0
    sharpen_alpha_upper = 1
    sharpen_lightness_lower = 0.75
    sharpen_lightness_upper = 1.5

    seq = iaa.Sequential([iaa.Sharpen(alpha=(sharpen_alpha_lower, sharpen_alpha_upper),
                                      lightness=(sharpen_lightness_lower, sharpen_lightness_upper)),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_sharpen_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        
          
def get_channel_image(image, n_images):
    add_lower = -10
    add_upper = 10

    seq = iaa.Sequential([iaa.Add((add_lower, add_upper), per_channel=0.5),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_channel_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        
        
def get_gray_image(image, n_images):
    grayscale_alpha_lower = 0.0
    grayscale_alpha_upper = 1.0

    seq = iaa.Sequential([iaa.Grayscale(alpha=(grayscale_alpha_lower, grayscale_alpha_upper))
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]

    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_grey_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        
        
def get_water_image(image, n_images):
    transformation_alpha = 50
    transformation_sigma = 9

    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=transformation_alpha, sigma=transformation_sigma),  
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_water_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        

def get_negative_image(image, n_images):
    saturation_lower = -40
    saturation_upper = 40

    seq = iaa.Sequential([iaa.AddToHueAndSaturation((saturation_lower, saturation_upper)),
                         ], random_order=True)

    images_aug = [seq.augment_image(image) for _ in range(n_images)]
    
    for i in range(0, len(images_aug)-1):
        imageio.imwrite(ROOT_DIR + DESTINATION_DIR                         
                        + f'image_negative_{str(uuid.uuid4().hex)}.jpg', images_aug[i])
        
        
def get_random_agumentation():
    # Constants Keras
    K = 50  # Number of agumentations
    rotation_range = 90
    width_shift_range=0.5
    height_shift_range=0.5
    shear_range=0.2
    zoom_range=0.2
    horizontal_flip=True

    # List all files
    array_files = os.listdir(ROOT_DIR + SOURCE_DIR)

    # Filtering out not .jpg files from the array
    array_images = [s for s in array_files if "jpg" in s]

    # Augmentation: Horizontal flip and rotation
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode='nearest')

    for image in array_images:
        # Load image and conver to array
        img = load_img(ROOT_DIR + SOURCE_DIR + image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        '''the .flow() command below generates batches of randomly transformed images
        and saves the results to the `preview/` directory'''
        i = 0
        for batch in datagen.flow(x,
                              batch_size=1,
                              save_to_dir=ROOT_DIR + DESTINATION_DIR,
                              save_prefix='random_augmented',
                              save_format='jpg'):
            i += 1
            if i > K:
                break


# List all rotated files
rotated_files = os.listdir(ROOT_DIR + ROTATED_DIR)

# Filtering out not .jpg files from the array
rotated_images = [s for s in rotated_files if "jpg" in s]

start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_gaussian_noise(image=image, n_images=10)

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_gaussian_noise: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_crop_images(image=image, n_images=10)

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_crop_images: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_flip_image(image=image, n_images=10)  

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_flip_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_blur_image(image=image, n_images=10) 
    
elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_blur_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_normalization_image(image=image, n_images=10) 

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_normalization_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_sharpen_image(image=image, n_images=10) 

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_sharpen_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')



start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_channel_image(image=image, n_images=10) 

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_channel_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_gray_image(image=image, n_images=10) 

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_gray_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_water_image(image=image, n_images=10) 

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_water_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    get_negative_image(image=image, n_images=10)

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_negative_image: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


start_time = time.time()

get_random_agumentation()

elapsed_time = time.time() - start_time
print(f'Elapsed Time - get_random_agumentation: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


# Complete function
start_time = time.time()

# Crop all images
for image in rotated_images:
    image = imageio.imread(ROOT_DIR + ROTATED_DIR + image)
    
    get_gaussian_noise(image=image, n_images=10)
    get_crop_images(image=image, n_images=10)
    get_flip_image(image=image, n_images=10)  
    get_blur_image(image=image, n_images=10) 
    get_normalization_image(image=image, n_images=10) 
    get_sharpen_image(image=image, n_images=10) 
    get_channel_image(image=image, n_images=10) 
    get_gray_image(image=image, n_images=10) 
    get_water_image(image=image, n_images=10) 
    get_negative_image(image=image, n_images=10)

get_random_agumentation()

elapsed_time = time.time() - start_time
print(f'Elapsed Time - Agumentation: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
