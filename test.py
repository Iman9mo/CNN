import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_phase_1_model(input_shape=(128, 128, 3), num_classes=10):
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_phase_2_model(phase_1_model, num_classes=10):
    """
    Adjusts the phase 1 model to create a phase 2 model with parameters maximized under 400,000.

    Args:
        phase_1_model (tf.keras.Model): Pretrained phase 1 model.
        num_classes (int): Number of output classes.

    Returns:
        phase_2_model (tf.keras.Model): Compiled phase 2 CNN model.
    """
    phase_2_model = models.Sequential()

    # Reuse all convolutional and pooling layers from phase_1_model, excluding dense layers
    for layer in phase_1_model.layers[:-4]:  # Skip the dense and dropout layers from phase 1
        phase_2_model.add(layer)

    # Add new Dense layers to maximize parameters within 400,000 limit
    phase_2_model.add(layers.Flatten())  # Ensure input is flattened for Dense layers
    phase_2_model.add(layers.Dense(8, activation='relu'))
    phase_2_model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    phase_2_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return phase_2_model

# Example usage
phase_1_model = create_phase_1_model()
phase_1_model.summary()
phase_2_model = create_phase_2_model(phase_1_model)
phase_2_model.summary()
