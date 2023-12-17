import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont

def create_character_image(character):
    # Create a blank image with white background
    image = Image.new('L', (30, 30), 0)
    draw = ImageDraw.Draw(image)

    # Define the font and size
    font = ImageFont.truetype('Roboto-MediumItalic.ttf', 26)

    # Draw the character
    draw.text((5, 0), character, fill=1, font=font)

    # Convert to numpy array and binarize
    data = np.array(image)
    data = (data > 0).astype(int)
    return data

# Characters to create images for
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:;,./<>=+-_'
character_images = {char: create_character_image(char) for char in characters}

# Prepare the input and output data
X = np.array([character_images[char] for char in characters])
y = np.array(list(range(len(characters))))  # Numeric labels

# Regenerate the character images and print their array representations
character_images = {char: create_character_image(char) for char in characters}

# Convert each character image to a string representation of a 2D array and print them
array_strings = {char: '\n'.join(' '.join(str(cell) for cell in row) for row in image)
                 for char, image in character_images.items()}

# Display the array representation for each character
for char in characters:  # Displaying all characters
    print(f"Character: {char}\n{array_strings[char]}\n")



# One-hot encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Create a simple neural network model
model = Sequential([
    Flatten(input_shape=(30, 30)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(characters), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
#print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


def predict_character(model, character_image, characters):
    # Preprocess the image (reshape to match model input)
    image_reshaped = character_image.reshape(1, 30, 30)

    # Predict the character
    prediction = model.predict(image_reshaped)
    predicted_index = np.argmax(prediction)

    return characters[predicted_index]


# Example: Predict a specific character
char_to_predict = '+'  # Change this to the character you want to predict
character_image = character_images[char_to_predict]

# Print the character array
print(f"Character Array for '\n{character_image}")

# Predict the character
predicted_char = predict_character(model, character_image, characters)
print(f"Predicted Character: '{predicted_char}'")

model.save('model.keras')
