import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.optimizers import Adam

def create_animal10_cnn_under_1_4M_params(
    input_shape=(128, 128, 3),
    num_classes=10,
    learning_rate=1e-3
):

    model = Sequential(name="Animals10_CNN_Under1_4M")

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # Block 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # Global Average Pooling instead of Flatten to reduce parameter count
    model.add(GlobalAveragePooling2D())

    # Dense block
    model.add(Dense(7000, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # Example usage:
    model = create_animal10_cnn_under_1_4M_params(
        input_shape=(128, 128, 3),
        num_classes=10,
        learning_rate=1e-3
    )
    model.summary()
