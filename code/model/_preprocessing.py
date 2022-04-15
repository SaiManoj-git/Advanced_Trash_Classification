from tensorflow.keras.preprocessing.image import ImageDataGenerator

# parser.add_argument('--output_dir', default='data/300x300_trashnetdata', help="Where to write the new data")


def preprocess(train_data_dir,dev_data_dir,test_data_dir, batch_size=64):

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,  # This is the source directory for training images
            target_size=(300, 300),  # All images will be resized to 300x300
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')


    # All images will be rescaled by 1./255
    dev_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    dev_generator = dev_datagen.flow_from_directory(
            dev_data_dir,  # This is the source directory for training images
            target_size=(300, 300),  # All images will be resized to 300x300
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')


    # All images will be rescaled by 1./255
    test_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    test_generator = test_datagen.flow_from_directory(
            test_data_dir,  # This is the source directory for training images
            target_size=(300, 300),  # All images will be resized to 300x300
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')

    inputs = {"train_datagen": train_datagen, "test_datagen":test_datagen, "dev_datagen":dev_datagen}
    return inputs