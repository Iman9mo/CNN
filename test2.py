import tensorflow as tf

def create_original_model(input_shape=(64, 64, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and summarize the original model
original_model = create_original_model()
original_model.summary()

def modify_dense_layers_for_400k(original_model, input_shape=(64, 64, 3), num_classes=10):
    # Rebuild the model with further reduced dense layers
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),  # Further reduced units
        tf.keras.layers.Dense(8, activation='relu'),  # Further reduced units
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# Modify the original model and summarize
modified_model = modify_dense_layers_for_400k(original_model)
modified_model.summary()


# Modify the original model and summarize
modified_model = modify_dense_layers_for_400k(original_model)
modified_model.summary()
