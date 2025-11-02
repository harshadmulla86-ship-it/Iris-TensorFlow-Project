# iris_model.py
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("\n🚀 Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model accuracy: {accuracy * 100:.2f}%")

# Make a prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
predicted_class = iris.target_names[np.argmax(prediction)]
print(f"🌸 Predicted class: {predicted_class}")

# 📊 Plot accuracy and loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Save the graph as an image file
plt.savefig("training_plot.png")
print("\n📸 Saved training graph as 'training_plot.png'")
plt.show()
