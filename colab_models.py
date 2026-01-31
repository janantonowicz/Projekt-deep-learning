import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

num_classes = 10
y_train = y_train.squeeze()
y_test = y_test.squeeze()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

class_names = ["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]

x_train[1].shape

BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

from sklearn.model_selection import train_test_split

# Podziel x_train i y_train na zbiory treningowy i walidacyjny
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Rozmiar nowego zbioru treningowego: {len(x_train_split)}")
print(f"Rozmiar zbioru walidacyjnego: {len(x_val)}")
print(f"Rozmiar zbioru testowego (pozostaje bez zmian): {len(x_test)}")

# Utwórz nowe tf.data.Dataset dla treningu i walidacji
train_ds_new = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split)).shuffle(50_000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)), # Jawnie określ kształt wejściowy jako 4D (wysokość, szerokość, kanały)
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1,),
    tf.keras.layers.RandomContrast(0.1,)
])

augmented_image = data_augmentation(x_train[0:1])
plt.imshow(augmented_image[0], cmap='grey')
plt.show()
augmented_image.shape

sample_img = x_train[0:1]

plt.figure(figsize=(10, 6))
for i in range(12):
    aug_img = data_augmentation(sample_img, training=True)
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(aug_img[0], cmap='grey')
    plt.axis("off")
plt.show()

def build_model(with_augmentation):
    # Fashion MNIST images are 28x28 grayscale, so input shape is (28, 28)
    inputs = tf.keras.Input(shape=(28, 28))
    # Add a channel dimension for grayscale images to make it (28, 28, 1)
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)

    if with_augmentation:
        x = data_augmentation(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"cnn_aug_{with_augmentation}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

with_augmentation = True

model_aug = build_model(with_augmentation= with_augmentation)
model_aug.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitoruj stratę walidacyjną
    patience=10,         # Czekaj 10 epok bez poprawy
    restore_best_weights=True # Przywróć najlepsze wagi
)

hist_aug = model_aug.fit(
    train_ds_new, # Użyj nowego zbioru treningowego
    validation_data=val_ds, # Użyj nowego zbioru walidacyjnego
    epochs=100, # Zwiększ liczbę epok, aby EarlyStopping mógł zadziałać
    callbacks=[early_stopping_callback], # Dodaj callback do treningu
    verbose=1,
)

acc = hist_aug.history['accuracy']
val_acc = hist_aug.history['val_accuracy']

loss = hist_aug.history['loss']
val_loss = hist_aug.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_loss, test_acc = model_aug.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

def saveModelToFile(WithAugmentation):
  if WithAugmentation:
    model_aug.save("model_with_aug.keras")
    model = tf.keras.models.load_model('model_with_aug.keras')
  else:
    model_aug.save('simple_fashion_mnist_model.keras')
    model = tf.keras.models.load_model('simple_fashion_mnist_model.keras')

saveModelToFile(with_augmentation)