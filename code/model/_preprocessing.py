from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', default='../data/300X300trashnetdata/train', help="Directory with the train data")
parser.add_argument('--dev_data_dir', default='../data/300X300trashnetdata/dev', help="Directory with the train data")
parser.add_argument('--test_data_dir', default='../data/300X300trashnetdata/test', help="Directory with the train data")

# parser.add_argument('--output_dir', default='data/300x300_trashnetdata', help="Where to write the new data")

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        args,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')