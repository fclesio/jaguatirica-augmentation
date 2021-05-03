# Jaguatirica Augmentation Wrapper
Wrapper for image augmentation for Deep Learning image classification tasks.

## How it works
The main purpose of this wrapper is to create heavy augmented image data 
for Deep Learning training.  

All images to be augmented should be placed 
in the `src/main/data/source` folder, and a minimum of 10 images is required. 

Due to that, every single image will be created at least 2.000 
augmented examples, all those files already split in the `train`, `test`,
 and `validation` folders. This will be discussed in the next section. 

## Preventing Data Leakage
In almost 99% of all posts and articles about Data Augmentation, we do 
not see the Data Leakage problem being discussed and this is a big issue 
for Deep Learning models. 

In this wrapper, there's a strict separation of the `train`, `test`,
 and `validation` folders to isolate those datasets to avoid Data Leakage that 
 will lead to overfitting. 

In simple terms the mechanism works in the following way:
- All images in the `reshaped` folder are included in an array and 
shuffled (using the `seed=42` for reproducibility);

- There's a fixed proportion for each set generated. For training, 
test and validation sets the proportion is 80%, 10%, and 10%;

- After this shuffle, all images receive the augmentation effect and
 are placed in their respective folder.

## Minimum Requirements
- Docker 19+
- Docker-Compose
 
## Reproducibility
All seeds has the value in ``42``, even for the libraries `imageio` and `imgaug`. 

## Performance
As we're using [``batch_augmentation``](https://imgaug.readthedocs.io/en/latest/source/api_multicore.html) module from `imgaug` library, by default 
all wrapper will run in multicore. Do not use python `multiprocessing` module due 
to the fact a child worker (i.e. different images) will be augmented more than 
once accidentally.

## Execution
### Data Placement
- Go to the folder `src/main/data/source` and delete all files contained 

- Still in the folder `src/main/data/source` include all files that you want 
to be augmented. The minimum amount of files that needs to be there is `10`


### Via command line
```bash
$ make && docker build -t sirius_image_augmentation . && docker-compose up 
```

## TO-DO
- [ ] Error handling
- [ ] Logging
- [ ] "Tree-shaking" code
 

# Why Jaguatirica (Ocelot in english)?
[Because it's a docile, night-wise and beatiful cat from South America](https://en.wikipedia.org/wiki/Ocelot). 
No special reason. In doubt? See [this video](https://www.youtube.com/watch?v=597LNt7HzCo).
