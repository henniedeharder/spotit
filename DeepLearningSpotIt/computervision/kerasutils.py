from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def load_cnn_model():
    return load_model('models/model.h5')

def get_predictions(directory, model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(directory,target_size=(400, 400),batch_size=20,class_mode='binary',shuffle=False)
    test_generator.reset()

    pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=0)
    predicted_probabilities = np.max(pred,axis=1)
    predicted_class_indices = np.argmax(pred,axis=1)

    return predicted_class_indices, predicted_probabilities