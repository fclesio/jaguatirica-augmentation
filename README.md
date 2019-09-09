# Jaguatirica Augmentation Wrapper
Wrapper for image augmentation for Deep Learning tasks regarding personal ID verification and other documents.

## Motivation
Generate data for Personal IDs can be a very hard task, especially to generate Deep Learning models. The objective with this tool is given a folder with some Personal IDs be able to generate at least 500 examples with slightly modified samples to have enough data to train Deep Learning algorithms.  Some IDs that can be used:
- [European IDs](https://en.wikipedia.org/wiki/National_identity_cards_in_the_European_Economic_Area)
- Cards from [Handwerkskammer](https://www.hwk-berlin.de/) ([Gewerbekarte and Handwerkskarte](https://www.deutsche-handwerks-zeitung.de/handwerks-und-gewerbekarten/150/3099/41407))
- [Aufenthaltstitel](https://de.wikipedia.org/wiki/Aufenthaltstitel) (German Blue Card)
- [Führerschein](https://de.wikipedia.org/wiki/F%C3%BChrerschein) (German Driver Licence)
- Passports
 
## Why don't you use the default Keras or common scripts for augmentation?
When someone needs to check physically one document there are several ways to assure if some document it's valid or not:3D holograms, textures, electronic verification with chip _et cetera_.

But for Deep Learning there's no way (yet) to pass such physical inputs and on top of that document images can vary a lot in terms of quality (_e.g._ blurry, contrast, color, brightness, _et cetera_).

The idea here it' generates files than can be readable for some human being without effort instead to generate a heavily augmented image, otherwise the DL network cannot learn from examples that can fool even a human being.

### It means that this wrapper won't generate a image that a human being cannot read?
Yes. We do not need DL networks learning from noise.

## Requirements
```
opencv-python==4.1.1.26
glob3==0.0.1
imageio==2.5.0
imgaug==0.2.9
Keras==2.2.5
numpy==1.17.1
Pillow==6.1.0
```
## Directory structure
```
ROOT
 - jaguatirica.py
   - destination
   - rotated
   - source
```

## TO-DO
- Error handling
- Include default folders
- Generate squared images with 244 x 244

# Why Jaguatirica (Ocelot in english)?
[Because it's a docile, night-wise and beatiful cat from South America](https://en.wikipedia.org/wiki/Ocelot). No special reason. In doubt? See [this video](https://www.youtube.com/watch?v=597LNt7HzCo).
