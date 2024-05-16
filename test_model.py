import tensorflow as tf
import os
import numpy as np
from keras.src.utils import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('modelA.keras')

# Specify the directory containing the images to be classified
image_dir = 'images'

# Specify the size of the images (should match the input shape of the model)
img_height, img_width = 28, 28

# Loop through all the images in the directory
for filename in os.listdir(image_dir):
    try:
        # Load the image using Keras' load_img function
        img = load_img(os.path.join(image_dir, filename), target_size=(img_height, img_width), color_mode='grayscale')

        # Convert the image to a numpy array
        img_array = img_to_array(img)

        # Invert the image by subtracting each pixel value from 255
        img_array = 255 - img_array

        # Reshape the image array to match the input shape of the model
        img_array = img_array.reshape((1, img_height, img_width, 1))

        # Normalize the pixel values to be between 0 and 1
        img_array = img_array / 255.0

        # Use the model to make a prediction on the image
        prediction = model.predict(img_array)

        # Get the predicted class (the class with the highest probability)
        predicted_class = np.argmax(prediction)

        # Generate a new filename with the predicted digit
        new_filename = f'{predicted_class}_{filename}'

        # Check if a file with the same name already exists
        if os.path.exists(os.path.join(image_dir, new_filename)):
            # If it does, append a number to the filename to make it unique
            i = 1
            while os.path.exists(os.path.join(image_dir, f'{new_filename}_{i}')):
                i += 1
            new_filename = f'{new_filename}_{i}'

        # Rename the file
        os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))
    except Exception as e:
        print(f'Error processing file {filename}: {str(e)}')