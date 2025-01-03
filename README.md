# Machine-Learning-in-Python
Python is one of the most popular programming languages for machine learning and it has replaced many languages in the industry, one of the reason is its vast collection of libraries. Python libraries that used in Machine Learning are:    Pandas,Matplotlib Numpy, seaborn, Scipy, Scikit-learn ,Natural Language Toolkit (NLTK),TensorFlow, Keras,PyTorch  

## [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
## [Machine Learning notes ](https://stanford.edu/~shervine/teaching/cs-229/)
## [Machine Learning Resources](https://drive.google.com/drive/folders/1KcE3sarMfwZyRNyT2pS5SPox4D0C8PWP)

https://rail.eecs.berkeley.edu/deeprlcourse/



#  Lightweight CNN Architecture

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
  

## **Usage**: Suitable for datasets with complex patterns and high feature variability.


# Deep Custom CNN Architecture


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

## usage: Ideal for smaller datasets or scenarios where computational efficiency is crucial.


### Pretrained ResNet50**

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the ResNet50 model with pre-trained weights
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Add custom layers
x = Flatten()(resnet_model.output)
output_layer = Dense(num_classes, activation="softmax")(x)

model_resnet = Model(inputs=resnet_model.input, outputs=output_layer)
model_resnet.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model_resnet.summary()
```

**Model Architecture**: 
- Deep residual learning framework with skip connections.

## **Usage**: - Excellent for transfer learning, allowing for rapid training on small datasets with good performance.


### Pretrained InceptionV3**

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the InceptionV3 model with pre-trained weights
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# Add custom layers
x = Flatten()(inception_model.output)
output_layer = Dense(num_classes, activation="softmax")(x)

model_inception = Model(inputs=inception_model.input, outputs=output_layer)
model_inception.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model_inception.summary()
```

**Model Architecture**: 
- Inception modules that allow for multiple filter sizes at each layer.

**Usage**: 
- Suitable for diverse image recognition tasks with varying object scales.


###  Pretrained VGG-16**

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the VGG-16 model with pre-trained weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Add custom layers
x = Flatten()(vgg_model.output)
output_layer = Dense(num_classes, activation="softmax")(x)

model_vgg = Model(inputs=vgg_model.input, outputs=output_layer)
model_vgg.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model_vgg.summary()
```

**Model Architecture**: 
- Sequential architecture with small filters and deep layers.

**Usage**: 
- Well-suited for high-resolution image classification tasks.

## 📊 **Comparison Table**

| Feature                 | Deep Custom CNN             | Lightweight CNN           | Pretrained ResNet50       | Pretrained InceptionV3    | Pretrained VGG-16         |
|-------------------------|-----------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| **Input Shape**         | Variable (depends on dataset)| (64, 64, 3)               | (224, 224, 3)             | (299, 299, 3)             | (224, 224, 3)             |
| **Layers**              | Deep Conv2D + MaxPooling    | Fewer Conv2D + MaxPooling | ResNet with skip connections| Inception modules         | Deep Conv2D               |
| **Trainable Parameters**| High                        | Low                       | High                       | High                      | High                      |
| **Transfer Learning**    | No                          | No                        | Yes                       | Yes                       | Yes                       |
| **Best For**           | Complex datasets            | Fast training on small datasets | Small datasets            | Diverse image recognition  | High-resolution tasks     |



