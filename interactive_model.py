import numpy as np
import cv2
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('modelA.keras')

# Initialize the window and the drawing variables
window_name = 'MNIST Digit Classifier'
drawing = False
ix, iy = -1, -1
img = np.zeros((512, 512, 1), np.uint8)  # Start with a black canvas

def draw(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), 255, 24)
            ix, iy = x, y
            display_prediction()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), 255, 24)
        display_prediction()

def display_prediction():
    # Resize to 28x28 and normalize
    small_img = cv2.resize(img, (28, 28))
    small_img = small_img.astype('float32') / 255.0
    small_img = small_img.reshape(1, 28, 28, 1)

    # Predict the digit
    preds = model.predict(small_img)
    sorted_indices = np.argsort(preds[0])[::-1]  # Get indices of sorted predictions
    top1 = sorted_indices[0]
    top2 = sorted_indices[1]
    confidences = tf.nn.softmax(preds).numpy()[0]  # Get softmax probabilities

    # Clear previous texts and redraw the image
    img[:50, :] = 0
    cv2.putText(img, f'Guess 1: {top1} Conf: {confidences[top1]:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'Guess 2: {top2} Conf: {confidences[top2]:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(window_name, img)

# Setup the window and callback
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

while True:
    cv2.imshow(window_name, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key to exit
        break
    elif k == ord('c'):  # Clear the canvas
        img = np.zeros((512, 512, 1), np.uint8)
        cv2.imshow(window_name, img)

cv2.destroyAllWindows()
