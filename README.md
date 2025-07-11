# Machine-Learning-in-Python
Python is one of the most popular programming languages for machine learning and it has replaced many languages in the industry, one of the reason is its vast collection of libraries. Python libraries that used in Machine Learning are:    Pandas,Matplotlib Numpy, seaborn, Scipy, Scikit-learn ,Natural Language Toolkit (NLTK),TensorFlow, Keras,PyTorch  

## [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
## [Machine Learning notes ](https://stanford.edu/~shervine/teaching/cs-229/)
## [Machine Learning Resources](https://drive.google.com/drive/folders/1KcE3sarMfwZyRNyT2pS5SPox4D0C8PWP)

### [Notes](https://rail.eecs.berkeley.edu/deeprlcourse/)

# Roadmap to Becoming an AI Engineer

This roadmap outlines the essential steps to become a proficient AI engineer, based on the skills and topics covered in a comprehensive Data Science, Machine Learning, and MLOps bootcamp. It is structured into four phases: Fundamentals, Machine Learning, Deep Learning, and Production & Deployment.

## Phase 1: Fundamentals
Build a strong foundation in programming, data manipulation, and databases.

- **Python Programming**
  - Learn Python fundamentals (variables, data structures, loops, functions).
  - Master data analysis libraries: NumPy, Pandas, Matplotlib.
  - Explore Python for data science and machine learning applications.
  - Reference: [Module 1: Python Fundamentals for DS and ML](https://www.udemy.com/course/module-1-python-fundamentals-for-ds-and-ml/)
  - Practice: Solve coding challenges on platforms like LeetCode or HackerRank.

- **Advanced Python for AI**
  - Study object-oriented programming, decorators, and generators.
  - Learn to optimize Python code for AI workflows.
  - Reference: [Module 2: Advanced Python for AI](https://www.udemy.com/course/module-2-advanced-python-for-ai/)

- **SQL for Data Science**
  - Understand relational databases and SQL queries.
  - Practice data extraction, filtering, and aggregation using MySQL.
  - Reference: [Module 3: MySQL for Data Science](https://www.udemy.com/course/module-3-my-sql-for-data-science/)

- **Career Preparation**
  - Develop a professional resume and optimize your LinkedIn profile.
  - Learn career guidelines specific to AI and data science roles.
  - Reference: Course module on "Resume and LinkedIn" and "Module 1: Career Guideline."

## Phase 2: Machine Learning
Gain expertise in machine learning algorithms and model building.

- **Core Machine Learning Concepts**
  - Study supervised and unsupervised learning algorithms (e.g., linear regression, decision trees, clustering).
  - Implement algorithms from scratch to understand their mechanics.
  - Use scikit-learn to build predictive models.
  - Reference: [Module 5: Machine Learning](#) and course section on "Implement machine learning algorithms from scratch."

- **Practical Machine Learning**
  - Work on real-world datasets to build and evaluate models.
  - Learn model evaluation metrics (accuracy, precision, recall, F1-score).
  - Practice hyperparameter tuning and cross-validation.
  - Reference: Course section on "Build predictive models using scikit-learn."

- **Big Data and Distributed Computing**
  - Explore tools like Apache Spark for handling large datasets.
  - Understand distributed computing concepts for scalability.
  - Reference: Course section on "Work with big data using distributed computing."

## Phase 3: Deep Learning
Dive into neural networks and advanced AI techniques.

- **Deep Learning Fundamentals**
  - Learn neural network architectures (e.g., CNNs, RNNs).
  - Understand frameworks like TensorFlow or PyTorch.
  - Reference: Course section on "Get familiar with Deep Learning."

- **Practical Deep Learning**
  - Build and train deep learning models for tasks like image classification or NLP.
  - Experiment with transfer learning and pre-trained models.
  - Practice optimizing neural networks (e.g., regularization, dropout).

- **Advanced Topics**
  - Explore generative AI models (e.g., GANs, VAEs).
  - Study reinforcement learning basics.
  - Reference: Course materials and external resources like DeepLearning.AI.

## Phase 4: Production & Deployment
Learn to deploy AI models and build end-to-end pipelines.

- **MLOps and Model Deployment**
  - Learn to deploy machine learning models to production using tools like Docker, Kubernetes, or Flask.
  - Understand CI/CD pipelines for AI workflows.
  - Reference: Course section on "Deploy machine learning models to production."

- **End-to-End Data Science Pipelines**
  - Build pipelines for data ingestion, preprocessing, model training, and inference.
  - Automate workflows using tools like Airflow or MLflow.
  - Reference: Course section on "Build end-to-end data science pipelines."

- **Version Control with Git and GitHub**
  - Master Git for version control and collaboration.
  - Learn to manage code repositories on GitHub.
  - Reference: [Module 9: Git and GitHub](#).**









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





# Autoencoder Implementation

### Key Features
- **Convolutional Encoder**: Uses convolutional layers to extract features and down-sample the input data into a latent representation.
- **Convolutional Decoder**: Uses transposed convolutional layers to reconstruct the input data from the latent representation.
- **Customizable Latent Dimension**: The latent space dimension can be adjusted to control the compression level.
- **Configurable Input Shape**: The model supports inputs of arbitrary dimensions (e.g., images with different sizes and channels).
- **MSE Loss**: The autoencoder minimizes the mean squared error (MSE) loss to optimize reconstruction quality.

## Model Architecture

### Encoder
The encoder consists of:
1. **Input Layer**: Accepts input data of specified shape.
2. **Convolutional Layers**: Extract features with increasing filters (32, 64, 128, 256) and ReLU activation.
3. **Max Pooling Layers**: Down-sample spatial dimensions.
4. **Flatten Layer**: Converts feature maps to a 1D vector.
5. **Dense Layer (Bottleneck)**: Compresses the features into a latent space representation of size `latent_dim`.

### Decoder
The decoder consists of:
1. **Input Layer**: Accepts latent space vectors.
2. **Dense Layer**: Expands the latent vector back into spatial dimensions.
3. **Reshape Layer**: Converts the expanded vector into feature maps.
4. **Transposed Convolutional Layers**: Reconstruct the input using filters (256, 128, 64, 32) and ReLU activation.
5. **Upsampling Layers**: Increase spatial dimensions back to the original input size.
6. **Output Layer**: Produces the final reconstructed image with a sigmoid activation.

### Autoencoder
The autoencoder combines the encoder and decoder into a single model:
- **Input**: Original data (e.g., images).
- **Output**: Reconstructed data.

## Model Specifications
- **Input Shape**: `(32, 32, 3)` (default; configurable).
- **Latent Dimension**: `128` (default; configurable).
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.

## Full Code
```python
from tensorflow.keras import layers, Model

def build_encoder(input_shape, latent_dim):
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)
    return Model(encoder_input, bottleneck, name="encoder")

def build_decoder(latent_dim, output_shape):
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(2 * 2 * 256, activation="relu")(decoder_input)
    x = layers.Reshape((2, 2, 256))(x)
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2DTranspose(output_shape[-1], (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)
    return Model(decoder_input, decoder_output, name="decoder")

def build_autoencoder(input_shape, latent_dim):
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    autoencoder_input = layers.Input(shape=input_shape, name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    autoencoder = Model(autoencoder_input, decoded, name="autoencoder")
    return autoencoder, encoder, decoder

# Define input shape and latent space dimension
input_shape = (32, 32, 3)
latent_dim = 128

# Build the autoencoder
autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_dim)

# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="mse")

# Display the model architecture
autoencoder.summary()
```

## Usage
### Building the Autoencoder
```python
# Define input shape and latent space dimension
input_shape = (32, 32, 3)
latent_dim = 128

# Build the autoencoder
autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_dim)

# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="mse")
```

### Training
```python
# Train the autoencoder
history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=64, validation_data=(x_val, x_val))
```

### Reconstruction
```python
# Encode and decode an image
encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)
```



# U-Net Implementation in TensorFlow/Keras

## Features
- **Fully Convolutional Network:** The model consists of an encoder, bottleneck, and decoder structure.
- **Skip Connections:** Uses `concatenate` to merge encoder and decoder layers for precise localization.
- **Binary Segmentation:** Outputs a single-channel mask with pixel values between 0 and 1.
- **Functional API:** Simplified implementation without object-oriented programming (OOP).

## Architecture
- **Encoder:** Repeated convolution and max-pooling layers to capture spatial features.
- **Bottleneck:** Dense feature representation at the narrowest part of the U.
- **Decoder:** Up-sampling with skip connections for accurate reconstruction.

![U-Net Architecture](https://miro.medium.com/max/1400/1*Z8dA1XKs8AFpIFHxFfp0pA.png)

### U-Net Implementation
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Down-sampling path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Up-sampling path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, (2, 2), activation='relu', padding='same')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, (2, 2), activation='relu', padding='same')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, (2, 2), activation='relu', padding='same')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = unet(input_size=(128, 128, 3))
model.summary()
```

Here's a GitHub README template for the provided neural network code:

---

# Simple Neural Network with Batch Normalization, Dropout, and Global Average Pooling 2D

## Features

- **Batch Normalization**: Stabilizes and accelerates training by normalizing the activations.
- **Dropout**: Reduces overfitting by randomly dropping neurons during training.
- **Global Average Pooling 2D**: Minimizes parameters by reducing each feature map to a single value.
- **Modular Design**: Easily adaptable for different datasets and tasks.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
def create_simple_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))         # Dropout

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))         # Dropout

    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))         # Dropout

    # Global Average Pooling 2D
    model.add(layers.GlobalAveragePooling2D())
    
    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())  # Batch Normalization
    model.add(layers.Dropout(0.5))          # Dropout

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Parameters
input_shape = (64, 64, 3)  # Example: 64x64 RGB images
num_classes = 10           # Example: 10 classes for classification

# Create and compile the model
model = create_simple_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()
```

## Explanation

1. **Input Layer**: Accepts input images of shape `(64, 64, 3)` (can be adjusted for your dataset).
2. **Convolutional Layers**:
   - Three convolutional layers with 32, 64, and 128 filters respectively.
   - Each convolutional layer is followed by Batch Normalization, MaxPooling, and Dropout.
3. **Global Average Pooling**: Aggregates the spatial dimensions of feature maps to a single value.
4. **Dense Layers**:
   - A fully connected layer with 128 neurons for feature extraction.
   - Batch Normalization and Dropout (50%) are applied.
5. **Output Layer**: A softmax layer for multi-class classification with `num_classes` outputs.


