import glob
import os

def get_dirs(path, classes):

    classes_dirs = []
    classes_videos = []
    for c in classes:

        classes_dirs.append([f for f in os.listdir(path + c) 
                    if os.path.isdir(os.path.join(path, c, f))])

        classes_dirs[-1].sort()

        classes_videos.append([])
        for f in classes_dirs[-1]:
            classes_videos[-1].append(path + c + '/' + f +
                               '/' + f + '.mp4')

        classes_videos[-1].sort()

    return classes_dirs, classes_videos


classes = ['Falls', 'NotFalls']
#path = '/mnt/hotstorage/Data/URFD-TRAIN/'
path = 'URFD/'

classes_dirs, classes_videos = get_dirs(path, classes) 

#OF_operation_list = ['invert', 'brigth', 'dark', 'sharp', 'dark_sharp', 'dropout', 'gauss_noise', 'salt', 'salt_pepper', 'contrast']
OF_operation_list = ['flip_h']

#pose_operation_list = ['invert', 'brigth', 'dark', 'sharp', 'dark_sharp', 'dropout', 'gauss_noise', 'salt', 'salt_pepper', 'contrast']
pose_operation_list = ['flip_h']

#frame_operation_list = ['invert', 'brigth', 'dark', 'sharp', 'dark_sharp', 'dropout', 'gauss_noise', 'salt', 'salt_pepper', 'contrast']
frame_operation_list = ['flip_h']

streams = ['frame', 'pose', 'flow_x', 'flow_y']
#streams = ['frame']

for c in range(len(classes)):
    for dir in classes_dirs[c]:
        print(dir)
        #frames = glob.glob(path + classes[c] + '/' + dir + '/frame_*.jpg') 
        #flow_x = glob.glob(path + classes[c] + '/' + dir + '/flow_x*.jpg') 
        #flow_y = glob.glob(path + classes[c] + '/' + dir + '/flow_y*.jpg') 
        #poses = glob.glob(path + classes[c] + '/' + dir + '/pose_*.jpg') 
        for s in streams:

            if s == 'frame':
                for op in OF_operation_list:
                    
                    #for f in frames:
                    #    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + f)
                    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + path + classes[c] + '/' + dir + '/frame_')
                
            elif s == 'pose':
                for op in OF_operation_list:

                    #for p in poses:
                    #    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + p)
                    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + path + classes[c] + '/' + dir + '/pose_')

            elif s == 'flow_x' or s == 'flow_y':
                for op in OF_operation_list:

                    #for f in flow_x:
                    #    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + f)
                    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + path + classes[c] + '/' + dir + '/flow_x_')

                    #for f in flow_y:
                    #    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + f)
                    os.system('python3 data_augmentation.py -op ' + op + ' -path ' + path + classes[c] + '/' + dir + '/flow_y_')
