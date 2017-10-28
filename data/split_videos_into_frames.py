try:
    import cv2
except:
    import opencv as cv2
import os
from glob import glob
import numpy as np

EXTENSIONS = ['mp4']

def main():
    
    for name in os.listdir('.'):
        frames_dir = '{}/frames'.format(name)
        videos_dir = '{}/videos/'.format(name)
        if not os.path.exists(videos_dir):
            continue

        # Get video filenames.
        video_filenames = []
        for extension in EXTENSIONS:
            video_filenames.extend(glob('{}/*.{}'.format(videos_dir, extension)))
        video_filenames = [filename.replace('\\','/') for filename in video_filenames]

        if not len(video_filenames):
            continue

        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        print(video_filenames)

        # Write a jpg image for every frame in each video file.
        for filename in video_filenames:
            vidcap = cv2.VideoCapture(filename)
            count = 0

            frames_dir_out = '{}/{}'.format(frames_dir, filename.replace(videos_dir,''))
            if not os.path.exists(frames_dir_out):
                os.makedirs(frames_dir_out)
            else:
                print("Directory already exists for video {}. Skipping this video.".format(filename))
                continue

            success = True
            while vidcap.isOpened() and success:
                success, image = vidcap.read()
                if success:
                    image = image.swapaxes(0,1)
                    image = np.flip(image,0)
                    cv2.imwrite('{}/frame_{}.jpg'.format(frames_dir_out, count), image)   
                    count += 1
            print('Read {} frames for filename {} and wrote them to {}'.format(count, filename, frames_dir_out))

if __name__ == '__main__':
    main()