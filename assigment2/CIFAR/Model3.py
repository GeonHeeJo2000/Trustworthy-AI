from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model

def Model3(input_tensor=None, load_weights=False):
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(32, 32, 3))
    x = Flatten()(base_model.output)
    x = Dropout(0.3)(x)  # Model2와 살짝 다르게
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax', name='before_softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    if load_weights:
        model.load_weights('./CIFAR/model3_cifar.h5')

    return model
