# Data Augmentation Tool

This code was built to create new data from already existent data, in order to
achieve better results training deep neural networks.
It's using python3, openCV3 and imgaug.

## Usage

For checking it's parameters you can run:

$ python3 data_augmentation.py

And an example of usage:

$ python3 change_resolution.py -op <operation> -path <image_path>

Important! Avaible data augmentation operations are listed in with $ python3 data_augmentation.py -help
Importan2! There are three avaible functions, to augmentate images or videos,
choose wisely changing the function call in the last lines of codes:
apply_video_transform, apply_image_transform, apply_multiple_image_transform.

The name of these functions should intuit what they actually do.

## Todo

Because of some openCV bug, it couldn't be used detected fourcc directly from
the video and instead was used cv2.VideoWriter_fourcc('M','J','P','G') as 
standard.  

In future, it could be the same as the original video, but only this coded
format was able to be saved after resizing.

## References

* https://github.com/aleju/imgaug
