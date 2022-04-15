"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


SIZE = 300

train_dev_test_split_size = [70,13,17]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/content/TDL_Project_Trash_Classification/code/data/trashnetdata', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='/content/TDL_Project_Trash_Classification/code/data/300X300_trashnetdata', help="Where to write the new data")


def resize_and_save(filename, output_dir, class_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(args.data_dir+"/"+class_dir+"/"+filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    cardboard_dir = os.path.join(args.data_dir,"cardboard")
    glass_dir = os.path.join(args.data_dir,"glass")
    metal_dir = os.path.join(args.data_dir,"metal")
    paper_dir = os.path.join(args.data_dir,"paper")
    plastic_dir = os.path.join(args.data_dir,"plastic")
    trash_dir = os.path.join(args.data_dir,"trash")


    # Get the filenames in each directory
    cardboardfilenames = os.listdir(cardboard_dir)
    glassfilenames = os.listdir(glass_dir)
    metalfilenames = os.listdir(metal_dir)
    paperfilenames = os.listdir(paper_dir)
    plasticfilenames = os.listdir(plastic_dir)
    trashfilenames = os.listdir(trash_dir)

    df = {"cardboard": cardboardfilenames, "glass": glassfilenames,"metal": metalfilenames,"paper": paperfilenames,"plastic": plasticfilenames, "trash": trashfilenames}

    # Split the images in 'train_signs' into 70/13/17 split
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    
    train_df={}
    dev_df={}
    test_df={}

    dev_test_df={}

    random.seed(230)
    # print(df)
    for key in df:
        random.shuffle(df[key])
            # 70/30
        split = int(train_dev_test_split_size[0]/100 * len(df[key]))
        print(key,"train-test-split:",split," df-len",len(df[key]),"\n")
        train_df[key] = df[key][:split]
        dev_test_df[key] = df[key][split:]

        # in 30: 43/67
        dev_test_split_size = train_dev_test_split_size[1]*100/(100-train_dev_test_split_size[0])
        dev_test_split = int(dev_test_split_size/100*len(dev_test_df[key])) 
        dev_df[key] = dev_test_df[key][:dev_test_split]
        test_df[key] = dev_test_df[key][dev_test_split:]




    
    filedirs = {'train': train_df,
                 'dev': dev_df,
                 'test': test_df}    

    
    # create output_dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
     # Preprocess train, dev and test
    for split in filedirs:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: output dir {} already exists".format(output_dir_split))
        
        for skey,sdir in filedirs[split].items():
            # print(skey," ",sdir)
            skey_dir_under_split = os.path.join(output_dir_split, '{}'.format(skey))
            if not os.path.exists(skey_dir_under_split):
                os.mkdir(skey_dir_under_split)

            for file in sdir:
                resize_and_save(file, skey_dir_under_split, skey, size=SIZE)


    print("Done building dataset")
