#%% md
# # Imports
#%%
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from keras.models import load_model
import random
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
#%% md
# # Constants
#%%
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 10

train_dir = './dataset/train'
val_dir = './dataset/val'
test_dir = './dataset/test'
#%% md
# # Function to create data generators
#%%
def create_data_generators(phase):
    if phase == 3:  # Data augmentation for Phase 3
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    else:  # Only scaling for Phase 1 and Phase 2
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, val_generator, test_generator
#%% md
# # Function to train a model
#%%
def train_model(
        model,
        train_gen,
        val_gen,
        epochs: int = 20,
        phase: int = 1
):
    callbacks = []
    if phase == 3:
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        callbacks = [early_stopping, reduce_lr]

        # model_checkpoint = ModelCheckpoint(
        #     f'phase_{phase}_best_model.h5',
        #     monitor='val_loss',
        #     save_best_only=True,
        #     verbose=1
        # )
        # callbacks.append(model_checkpoint)

    # Train the model with or without callbacks
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    return history
#%% md
# # Function to plot learning curves
#%%
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
#%% md
# # Function to evaluate the model
#%%
def evaluate_model(model, test_gen):
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

    y_true_binary = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(12, 8))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {list(test_gen.class_indices.keys())[i]} (AUC: {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
#%% md
# # visualize predictions
#%%
def visualize_random_predictions(model, test_gen, num_images=10):
    x_test, y_test = next(test_gen)
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Randomly select indices
    random_indices = random.sample(range(len(x_test)), min(num_images, len(x_test)))

    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(np.squeeze(x_test[idx]))  # Squeeze in case of single-channel images
        true_label = list(test_gen.class_indices.keys())[y_true_classes[idx]]
        pred_label = list(test_gen.class_indices.keys())[y_pred_classes[idx]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
#%% md
# # Phase 1: Model with 1,400,000 parameters (no data augmentation)
#%%
def create_phase_1_model(
        input_shape=(128, 128, 3),
        num_classes=10,
        learning_rate=1e-3
):
    model = Sequential(name="Animals10_CNN_Under1_4M")

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Block 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Global Average Pooling instead of Flatten to reduce parameter count
    model.add(GlobalAveragePooling2D())

    # Dense block
    model.add(Dense(7000, activation='relu'))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model
#%% md
# # Phase 2: Adjusted model with 400,000 parameters (reuse Phase 1 architecture)
#%%
def create_phase_2_model(
        input_shape=(128, 128, 3),
        num_classes=10,
        learning_rate=1e-3
):
    model = Sequential(name="Animals10_CNN_Under_400k")

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Block 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Global Average Pooling instead of Flatten to reduce parameter count
    model.add(GlobalAveragePooling2D())

    # Dense block
    model.add(Dense(512, activation='relu'))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model
#%%
def create_phase_3_model_1(
        input_shape=(128, 128, 3),
        num_classes=10
):
    model = Sequential(name="phase_3_model_1")

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Block 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Global Average Pooling instead of Flatten to reduce parameter count
    model.add(GlobalAveragePooling2D())

    # Dense block
    model.add(Dense(7000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    return model
#%%
def create_phase_3_model_2(
        input_shape=(128, 128, 3),
        num_classes=10
):
    model = Sequential(name="phase_3_model_2")

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Block 4
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Global Average Pooling instead of Flatten to reduce parameter count
    model.add(GlobalAveragePooling2D())

    # Dense block
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    return model
#%% md
# # Phase 3: Overfitting prevention
#%%
def apply_overfitting_prevention(model):
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            layer.kernel_regularizer = regularizers.l2(0.001)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
#%% md
# # Execution for Phase 1
#%%
train_generator, val_generator, test_generator = create_data_generators(phase=1)
#%%
model_phase_1 = create_phase_1_model()
#%%
model_phase_1.summary()
#%%
history_phase_1 = train_model(model_phase_1, train_generator, val_generator, phase=1)
#%%
plot_learning_curves(history_phase_1)
#%%
evaluate_model(model_phase_1, test_generator)
#%%
visualize_random_predictions(model_phase_1, test_generator)
#%% md
# # Execution for Phase 2
#%%
model_phase_2 = create_phase_2_model()
#%%
model_phase_2.summary()
#%%
history_phase_2 = train_model(model_phase_2, train_generator, val_generator, phase=2)
#%%
plot_learning_curves(history_phase_2)
#%%
evaluate_model(model_phase_2, test_generator)
#%%
visualize_random_predictions(model_phase_2, test_generator)
#%% md
# # Execution for Phase 3: Prevent Overfitting (Phase 1 and Phase 2 models)
#%%
train_generator, val_generator, test_generator = create_data_generators(phase=3)
#%%
model_phase_3_1 = create_phase_3_model_1()
model_phase_3_1 = apply_overfitting_prevention(model_phase_3_1)
history_phase3_1 = train_model(model_phase_3_1, train_generator, val_generator, phase=3)
model_phase_3_1.save("phase_3_model_1.keras")
#%%
plot_learning_curves(history_phase3_1)
#%%
evaluate_model(model_phase_3_1, test_generator)
#%%
visualize_random_predictions(model_phase_3_1, test_generator)
#%%
model_phase_3_2 = create_phase_3_model_2()
model_phase_3_2 = apply_overfitting_prevention(model_phase_3_2)
history_phase3_2 = train_model(model_phase_3_2, train_generator, val_generator, phase=3)
model_phase_3_2.save("phase_3_model_2.keras")
#%%
plot_learning_curves(history_phase3_2)
#%%
evaluate_model(model_phase_3_2, test_generator)
#%%
visualize_random_predictions(model_phase_3_2, test_generator)