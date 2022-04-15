
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train(model, inputs, epochs=30,batch_size=64, pickle_name="mymodel"):
    nb_train_samples = 1766
    nb_validation_samples = 327
    nb_test_samples = 434
    

    #Callback to save the best model
    callbacks_list = [
        ModelCheckpoint(filepath=f'pickle/{pickle_name}.h5',monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10,verbose=1)
    ]

    #Training
    history = model.fit(
            inputs["train_datagen"],
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            callbacks = callbacks_list,
            validation_data=inputs["dev_datagen"],
            verbose = 1,
            validation_steps=nb_validation_samples // batch_size)

    return history