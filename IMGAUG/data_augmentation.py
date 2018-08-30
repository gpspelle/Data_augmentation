from imgaug import augmenters as iaa
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import sys


def apply_video_transform(transform, path):
    
    cap = cv.VideoCapture(path)
    fps = int(round(cap.get(cv.CAP_PROP_FPS)))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    seq = transform_op(transform)

    if seq == None:
        return

    # So far it's hardcoded for '.mp4' format
    # Need to remove everything after '.' to add transform to it's name
    if path[-4:] != '.mp4':
        print("Sorry, only .mp4 is accepted, but that's not hard to fix")
        return 
    
    path = path[:-4] + '_' + transform + '.mp4'

    out = cv.VideoWriter(path, cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

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

def transform_op(transform):
    
    if transform == 'flip_h':
        seq = iaa.Sequential([
             iaa.Fliplr(1),
        ])
    elif transform == 'invert':
        seq = iaa.Sequential([
             iaa.Invert(1, True),
        ])
    elif transform == 'brigth':
        seq = iaa.Sequential([
             iaa.Multiply(1.5, True),
        ])
    elif transform == 'dark':
        seq = iaa.Sequential([
             iaa.Multiply(0.5, True),
        ])
    elif transform == 'blur':
        seq = iaa.Sequential([
             iaa.GaussianBlur(0.5),
        ])
    elif transform == 'sharp':
        seq = iaa.Sequential([
             iaa.Sharpen(1, 1.25),
        ])
    elif transform == 'dark_sharp':
        seq = iaa.Sequential([
             iaa.Sharpen(1, 0.25),
        ])
    elif transform == 'gauss_noise':
        seq = iaa.Sequential([
             iaa.AdditiveGaussianNoise(0.03, 10, True),
        ])
    elif transform == 'dropout':
        seq = iaa.Sequential([
             iaa.Dropout(0.08, True),
        ])
    elif transform == 'salt':
        seq = iaa.Sequential([
             iaa.Salt(0.08, True),
        ])
    elif transform == 'salt_pepper':
        seq = iaa.Sequential([
             iaa.SaltAndPepper(0.08, True),
        ])
    elif transform == 'contrast':
        seq = iaa.Sequential([
             iaa.ContrastNormalization(1.5, True),
        ])
    else:
        # So far it's only implemented horizontal flips 
        print("Sorry, only those operations listed in help are implemented. \
                Check data_augmentation.py -help for further instructions")
        return None

    return seq

def apply_image_transform(transform, path): 
    
    img = np.asarray(cv.imread(path))

    seq = transform_op(transform)

    if seq == None:
        return

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
    
    pos = path.rfind('/')
    end_string = path[pos+1:]

    print(end_string)

    new_end_string = transform + '_' + end_string
    new_path = path[:pos] + '/' + new_end_string
    
    #path = path[:-4] + '_' + transform + '.jpg'

    cv.imwrite(new_path, img_aug)
    #plt.axis("off")
    #img = plt.imshow(cv.cvtColor(img_aug, cv.COLOR_BGR2RGB))
    #plt.show()

def apply_multiple_image_transform(transform, path): 
    
    seq = transform_op(transform)

    if seq == None:
        return

    frames_path = glob.glob(path + '*.jpg') 
    images = []

    for image in frames_path:
        img = np.asarray(cv.imread(image))
        images.append(img)

    images_aug = seq.augment_images(images)

    for img_aug, path in zip(images_aug, frames_path):
        # So far it's hardcoded for '.jpg' format
        # Need to remove everything before '.' to add transform to it's name
        if path[-4:] != '.jpg':
            print("Sorry, only .jpg is accepted, but that's not hard to fix")
            return


        pos = path.rfind('/')

        second_pos = path[:pos-1].rfind('/')
        end_string = path[second_pos+1:]
        
        if not os.path.exists(path[:second_pos] + '/' + transform + '_' + path[second_pos+1:pos]):
            os.makedirs(path[:second_pos] + '/' + transform + '_' + path[second_pos+1:pos])

        new_path = path[:second_pos] + '/' + transform + '_' + end_string

        #path = path[:-4] + '_' + transform + '.jpg'
        cv.imwrite(new_path, img_aug)

if __name__ == '__main__':
    #print("***********************************************************",
    #        file=sys.stderr)
    #print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    #print("***********************************************************",
    #        file=sys.stderr)

    argp = argparse.ArgumentParser(description='Data Augmentation Tool')
    argp.add_argument("-path", dest="path", type=str, nargs=1,
                     help="Usage: -path <path_to_video>", required=True)
    argp.add_argument("-op", dest="op", type=str, nargs=1,
                     help="Usage: -op flip_h"  + "\n" +
                             "Avaible ops       : " + '\n' +
                             "flip_h            : 50%% horizontal flip \
                              invert            : pixel = 1 - pixel \
                              bright            : pixel = 1.5 * pixel \
                              dark              : pixel = 0.5 * pixel \
                              blur              : guassian blur \
                              sharp             : light sharp \
                              dark_sharp        : dark sharp \
                              gauss_noise       : gaussian noise \
                              dropout           : black noise \
                              salt              : white noise \
                              salt and pepper   : white and black noise \
                              contrast          : contrast normalization \
                             ", required=True)
    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    apply_multiple_image_transform(args.op[0], args.path[0])
