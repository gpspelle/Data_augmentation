from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import sys


def apply_video_transform(transform, path):

    if transform == 'flip_h':
        seq = iaa.Sequential([
             iaa.Fliplr(1),
        ])
    else:

        # So far it's only implemented horizontal flips 
        print("Sorry, only 'flip_h' is implemented yet. \
                Check data_augmentation.py -help for further instructions")
        return

    cap = cv.VideoCapture(path)
    fps = int(round((cv.VideoCapture(path)).get(cv.CAP_PROP_FPS)))

    # So far it's hardcoded for '.mp4' format
    # Need to remove everything after '.' to add transform to it's name
    if path[-4:] != '.mp4':
        print("Sorry, only .mp4 is accepted, but that's not hard to fix")
        return 
    
    path = path[:-4] + '_' + transform + '.mp4'

    out = cv.VideoWriter(path, cv.VideoWriter_fourcc('M','J','P','G'), fps, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
    
        sucess, frame = cap.read()
        if sucess == False:
            break

        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
   
        images = []
        images.append(frame)
        images_aug = seq.augment_images(images)

        out.write(images_aug[0])

    cap.release()
    out.release()
    cv.destroyAllWindows()

def apply_image_transform(transform, path): 

    if transform == 'flip_h':
        seq = iaa.Sequential([
             iaa.Fliplr(1),
        ])
    else:

        # So far it's only implemented horizontal flips 
        print("Sorry, only 'flip_h' is implemented yet. \
                Check data_augmentation.py -help for further instructions")
        return

    img = np.asarray(cv.imread(path))

    images = []

    images.append(img)
    images_aug = seq.augment_images(images)

    #transformed_images = seq.augment_images(images)
    #for img in transformed_images:

    img_aug = images_aug[0]

    # So far it's hardcoded for '.jpg' format
    # Need to remove everything before '.' to add transform to it's name
    if path[-4:] != '.jpg':
        print("Sorry, only .jpg is accepted, but that's not hard to fix")
        return
    
    path = path[:-4] + '_' + transform + '.jpg'

    cv.imwrite(path, img_aug)
    #plt.axis("off")
    #img = plt.imshow(cv.cvtColor(img_aug, cv.COLOR_BGR2RGB))
    #plt.show()

if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)

argp = argparse.ArgumentParser(description='Data Augmentation Tool')
argp.add_argument("-path", dest="path", type=str, nargs=1,
                 help="Usage: -path <path_to_video>", required=True)
argp.add_argument("-op", dest="op", type=str, nargs=1,
                 help="Usage: -op flip_h"  + "\n" +
                         "Avaible ops: " + '\n' +
                         "flip_h : 50%% horizontal flip", required=True)
try:
    args = argp.parse_args()
except:
    argp.print_help(sys.stderr)
    exit(1)

apply_video_transform(args.op[0], args.path[0])
