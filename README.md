
# MNIST Digit Classifier

This project demonstrates how to build, train, and use a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The project includes scripts for training the model, testing the model with images, and an interactive drawing interface to classify digits in real-time.

## Files in the Project

- **`model.py`**: Script to train and save a CNN model on the MNIST dataset.
- **`test_model.py`**: Script to test the saved model on images from a specified directory.
- **`interactive_model.py`**: Script to interactively draw and classify digits in real-time using the saved model.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- OpenCV (for `interactive_model.py`)

## Setup

1. **Install the required packages**:
   ```sh
   pip install tensorflow numpy opencv-python
   ```

2. **Download the MNIST dataset**: The dataset is automatically downloaded when running `model.py`.

## Usage

### Training the Model

To train the CNN model on the MNIST dataset, run the `model.py` script:

```sh
python model.py
```

This will train the model and save it as `modelA.keras`.

### Testing the Model

To test the saved model on images from a directory, run the `test_model.py` script:

1. Create a directory named `images` and place the images you want to classify in it.
2. Run the script:
   ```sh
   python test_model.py
   ```

The script will output the predicted digit for each image in the `images` directory.

### Interactive Digit Classification

To use the interactive drawing interface to classify digits in real-time, run the `interactive_model.py` script:

```sh
python interactive_model.py
```

A window will open where you can draw digits using the mouse. The model will classify the drawn digit in real-time and display the top predictions with their confidence levels.

## Project Structure

- **`model.py`**: Contains code for loading and preprocessing the MNIST dataset, defining the CNN architecture, training the model, and saving it.
- **`test_model.py`**: Contains code for loading the saved model, preprocessing images from a directory, and outputting predictions.
- **`interactive_model.py`**: Contains code for creating an interactive drawing window, capturing mouse events, preprocessing the drawn digit, and displaying predictions in real-time.
