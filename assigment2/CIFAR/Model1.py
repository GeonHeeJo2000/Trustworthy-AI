'''
ResNet50 model for CIFAR-10 classification
'''

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

def Model1(input_tensor=None, load_weights=False):
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(32, 32, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax', name='before_softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    if load_weights:
        model.load_weights('./CIFAR/model1_cifar.h5')

    return model