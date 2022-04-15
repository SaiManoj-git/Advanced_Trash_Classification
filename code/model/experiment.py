from _preprocessing import preprocess

from _train import train

from _model import simpleCNN

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', default='../data/300X300_trashnetdata/train', help="Directory with the train data")
parser.add_argument('--dev_data_dir', default='../data/300X300_trashnetdata/dev', help="Directory with the train data")
parser.add_argument('--test_data_dir', default='../data/300X300_trashnetdata/test', help="Directory with the train data")

args = parser.parse_args()

inputs = preprocess(args.train_data_dir, args.dev_data_dir, args.test_data_dir)
# print(inputs["train_datagen"])

# classes = list(inputs["train_datagen"].class_indices.keys())
# print('Classes: '+str(classes))
# num_classes  = len(classes)
model = simpleCNN(300,6)

history = train(model, inputs, 30,64, "testmodel")

