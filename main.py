import data.split_videos_into_frames
import os

def setup():
    os.chdir('data')
    data.split_videos_into_frames.main()
    os.chdir('..')

def main():
    setup()

if __name__ == '__main__':
    main()