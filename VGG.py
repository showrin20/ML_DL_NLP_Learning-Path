from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Assuming `X_train` and `classes` are already defined
input_shape = X_train[0].shape
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

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]  # Fixed the typo in "metrics"
)

# Print model summary
model.summary()  # Fixed the typo in "summary"
