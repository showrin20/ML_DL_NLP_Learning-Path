# Machine-Learning-in-Python
Python is one of the most popular programming languages for machine learning and it has replaced many languages in the industry, one of the reason is its vast collection of libraries. Python libraries that used in Machine Learning are:    Pandas,Matplotlib Numpy, seaborn, Scipy, Scikit-learn ,Natural Language Toolkit (NLTK),TensorFlow, Keras,PyTorch  

## [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
## [Machine Learning notes ](https://stanford.edu/~shervine/teaching/cs-229/)
## [Machine Learning Resources](https://drive.google.com/drive/folders/1KcE3sarMfwZyRNyT2pS5SPox4D0C8PWP)

https://rail.eecs.berkeley.edu/deeprlcourse/



# Simple Convolutional Neural Network (CNN)

```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define input shape and number of classes
input_shape = (64, 64, 3)  # Example: 64x64 RGB images
num_classes = 10  # Example: 10 classes

# Define the model
input_layer = Input(shape=input_shape)

# Convolutional and pooling layers
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Flatten and fully connected layers
x = Flatten()(x)
x = Dense(128, activation="relu")(x)

# Output layer
output_layer = Dense(num_classes, activation="softmax")(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model summary
model.summary()
```

1. **Conv2D Layers**:
   - The first convolutional layer has 32 filters and uses a 3x3 kernel size.
   - The second convolutional layer doubles the filters to 64 for deeper feature extraction.

2. **MaxPooling2D Layers**:
   - Pooling layers reduce spatial dimensions and computational complexity.

3. **Dense Layer**:
   - A fully connected layer with 128 units acts as the final feature abstraction before the output layer.

4. **Output Layer**:
   - A `Dense` layer with `num_classes` units and a `softmax` activation function for classification.
  


# CNN Model for Image Classification


## 🧩 Model Architecture : The model is inspired by VGG-style CNN architectures and consists of:

1. **Input Layer**: Accepts input images of specified shape.
2. **Convolutional Layers**: 
   - Extract spatial features using 3x3 filters.
   - Employ ReLU activation for non-linearity.
   - Use `padding="same"` to maintain spatial dimensions.
3. **Max Pooling Layers**:
   - Downsample feature maps using 2x2 pooling.
4. **Fully Connected Layers**:
   - Flatten the feature maps.
   - Two dense layers with 4096 neurons each.
5. **Output Layer**:
   - Dense layer with `len(classes)` neurons.
   - Uses a softmax activation function to output probabilities for each class.


### Import Libraries
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
```

### Define the Model
```python
# Input layer
input_layer = Input(shape=input_shape)

# Convolutional and pooling layers
x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

# Fully connected layers
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dense(4096, activation="relu")(x)

# Output layer
output_layer = Dense(len(classes), activation="softmax")(x)

# Model definition
model = Model(inputs=input_layer, outputs=output_layer)
```

### Compile the Model
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Model Summary
```python
model.summary()
```
