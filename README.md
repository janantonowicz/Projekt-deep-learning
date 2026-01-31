# Fashion MNIST Image Classification

### Python Web & Deep Learning Project Report

## 🧠 Task Description

The goal of this project is **image classification** using deep learning. The neural network assigns each input image to one of **10 categories**:

`t-shirt`, `trouser`, `pullover`, `dress`, `coat`, `sandal`, `shirt`, `sneaker`, `bag`, `ankle boot`

## 📊 Dataset Description

### Source

The project uses the **Fashion MNIST** dataset.

### Dataset Properties

- 70,000 grayscale images
    
- Image size: **28 × 28 pixels**
    
- Each image belongs to one of 10 classes
    
- Data is normalized to the range **[0, 1]**
    
- Images reshaped to **28 × 28 × 1** (added depth channel)
    

### Dataset Split

|Set|Samples|Percentage|
|---|---|---|
|Training|48,000|68.5%|
|Validation|12,000|17%|
|Test|10,000|14.5%|

## 🏗️ Model Architecture & Training Process

### 🔹 Convolutional Blocks

**Block 1**

- 2 × Conv2D (32 filters, kernel 3×3)
    
- MaxPooling2D
    

**Block 2**

- 2 × Conv2D (64 filters, kernel 3×3)
    
- MaxPooling2D
    

Convolutional layers scan the image for patterns.

- Block 1 learns ~32 simple features (lines, dots, edges)
    
- Block 2 learns ~64 more complex patterns
    

### 🔹 Dense Layers

After convolutional blocks:

python

```
X = tf.keras.layers.Flatten()(X)
X = tf.keras.layers.Dense(128, activation="relu")(X)
```

- Hidden layer: **128 neurons**, ReLU activation
    
- Output layer: **10 neurons**, Softmax activation (one per class)
    

### 🔹 Loss Function

**Sparse Categorical Crossentropy** Used for multi-class classification with integer labels.

### 🔹 Optimizer

python

```
optimizer = tf.keras.optimizers.Adam(1e-3)
```

Adam adjusts network weights based on the loss to improve predictions.

### 🔹 Batch Size

Kod

```
BATCH_SIZE = 128
```

Weights are updated after every 128 images.

### 🔹 Early Stopping

To prevent overfitting:

- `patience = 10`
    
- Training stops if validation loss does not improve for 10 epochs
    
- `restore_best_weights = True` ensures the best model is saved
    

## 🖼️ Model with Data Augmentation

Applied augmentations:

- Random horizontal flip
    
- Random translation
    
- Random rotation
    
- Random zoom
    
- Random contrast adjustment
    

Augmentation increases dataset diversity and reduces overfitting by preventing the model from memorizing the training set.

## 📈 Model Comparison

### Augmented Model

<img width="945" height="357" alt="image" src="https://github.com/user-attachments/assets/257d1165-df07-477e-80c6-9dc8b6315507" />


```
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9054 - loss: 0.2717
Test accuracy: 0.9064
```

### Simple Model

<img width="945" height="354" alt="image" src="https://github.com/user-attachments/assets/9fb6dc12-c0e9-40ff-95cd-fc2fb61968c8" />


```
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9231 - loss: 0.2275
Test accuracy: 0.9226
```

### Analysis

- The **simple model achieved ~1.5% higher test accuracy**.
    
- Both models show **overfitting**, although early stopping helped reduce it.
    
- The augmented model required more epochs but generalizes better in theory.
    
- Both models struggle with images that differ from the training distribution (e.g., different background colors or higher resolution).
    
- Since the model expects **28×28** input, larger images must be downscaled, which may remove important features.
    

## 📝 Summary

This project demonstrates the full workflow of building a convolutional neural network for image classification using Fashion MNIST. Both a baseline model and an augmented model were trained and evaluated. While the simple model achieved slightly higher accuracy, both models exhibit limitations when applied to images outside the dataset's narrow domain.
