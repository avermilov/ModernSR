# ModernSR

## Overview
This is a small custom deep learning x2/x4 Video/Image Super Resolution framework that I wrote during my second year at uni, designed to be easy to use and customizable.
You can use it to train, validate, test and inference your Super Resolution models, as well as do a bit of data preparation and data viewing. **NOTE: this framework does not use GANs.**

## Models
There are several models available for training:
[SR CNN](https://arxiv.org/abs/1501.00092), [SR ResNet](https://arxiv.org/abs/1609.04802), [Residual Dense Network](https://arxiv.org/abs/1802.08797), [Residual in Residual Dense Block](https://arxiv.org/abs/1809.00219). Model code was collected from other repositories and was not created by me (duh!). Recommended architecture is Residual Dense Network.

## LowRes-HighRes Image Pair Creation
To create pair images for training data, you can use several approaches:

- Simplest option: use interpolation to downscale an image. However, it will most likely also be the most useless one. 

- More advanced: use kernels from previously extracted images similar to your target domain ones to create a LowRes image from your HighRes one. To generate the kernels, use the [KernelGAN Repository](https://github.com/sefibk/KernelGAN). The quality will most likely be much better.
- Most adavnced: use aforementioned kernels + noise patches extracted from your target domain images to apply noise to LowRes images after downscaling them. To extract noise images from your target domain images, use the utils/extract_noise.py script.

## Training
Use the train.py script to train, validate and test your models. To begin training, pass a path to a config file to be used:
```
python3 train.py example_config.cfg
```

## Inference
Use the inference.py script to upscale images using a trained model. You can either pass a config with an inference section or specify all required arguments through the command line. **NOTE: Keep in mind that these models are quite big, so you will require a beefy GPU with enough VRAM to inference faster.**
```
sage: inference.py [-h] [--config CONFIG] [--in_dir IN_DIR]
                    [--out_dir OUT_DIR] [--workers WORKERS]
                    [--batch_size BATCH_SIZE] [--scale SCALE]
                    [--checkpoint_path CHECKPOINT_PATH] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file for training Super Resolution models.
  --in_dir IN_DIR       Input inference image directory.
  --out_dir OUT_DIR     Inference results directory.
  --workers WORKERS     Number of workers for loading inference dataset.
  --batch_size BATCH_SIZE
                        Batch size of inference data loader.
  --scale SCALE         Super Resolution scale of the model.
  --checkpoint_path CHECKPOINT_PATH
                        Path to checkpoint to be used for inference.
  --model MODEL         Model type to be used for inference.
```

## Helper Scripts
There are several helper scripts available to make the framework easier to use.
### concat_videos.py
This script allows you to concatenate two videos together vertically or horizontally (passed as a parameter). You can use it to compare the effectiveness of the models used to create the videos.
```
usage: concat_videos.py [-h] [--stack {h,v}] vid1 vid2

positional arguments:
  vid1           First video.
  vid2           Second video.

optional arguments:
  -h, --help     show this help message and exit
  --stack {h,v}  Stack videos vertically or horizontally.
```
### extract_noise.py
This script extracts noise from images that you can use for training your models. Alternatively, you can use it to get denoised images.
```
usage: extract_noise.py [-h] [--src_dir SRC_DIR] [--dest_dir DEST_DIR]
                        [--noise_level NOISE_LEVEL]
                        [--window_size WINDOW_SIZE]
                        [--blur_kernel_size BLUR_KERNEL_SIZE]
                        [--operation OPERATION]

optional arguments:
  -h, --help            show this help message and exit
  --src_dir SRC_DIR     Image directory from which content is to be used.
  --dest_dir DEST_DIR   Directory in which to save the results.
  --noise_level NOISE_LEVEL
                        How strong extracted noise should be.
  --window_size WINDOW_SIZE
                        Size of window to be used for computing average.
  --blur_kernel_size BLUR_KERNEL_SIZE
                        Size of blur kernel to be used.
  --operation OPERATION
                        Operation to be done: extract noise or denoise images.
```

### make_video_dataset.py
This script allows you to create a dataset from a video or a movie by sampling and saving frames with a given frequency.
```
usage: make_film_dataset.py [-h] [--prefix PREFIX] src_path dest_dir frequency

positional arguments:
  src_path         Path to film video to be used.
  dest_dir         Directory in which results are to be saved.
  frequency        How often (in frames) a frame should be saved.

optional arguments:
  -h, --help       show this help message and exit
  --prefix PREFIX  Image prefix.
```

### split_train_val.py
This script allows you to easily split a data folder into train and validation folders with a given train share. 
Example: image_dir/ --> image_dir/train/, image_dir/valid/
```
usage: split_train_val.py [-h] src_dir train_share

positional arguments:
  src_dir      Noise or kernels to be moved
  train_share  Percentage of files to be used for training

optional arguments:
  -h, --help   show this help message and exit
```

### visualize_kernels.py
This script allows you to visualize extracted image kernels for analysis or filtering.
```
usage: visualize_kernels.py [-h] src_dir dest_dir

positional arguments:
  src_dir     Source kernels directory to be used.
  dest_dir    Directory in which resulting images are to be put.

optional arguments:
  -h, --help  show this help message and exit
```
