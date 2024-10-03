# prompt: código python para, usando o pacote opencv, capturar imagem da webcam e identificar números na imagem capturada, usando o modelo "model", construído anteriormente

import cv2
import numpy as np
import pickle
import argparse



def predict_digit_from_webcam(model):
  """Captures image from webcam, identifies digits using the provided model, and displays the result."""

  # Initialize webcam
  cap = cv2.VideoCapture(0)

  while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to isolate digits
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      # Get bounding rectangle of contour
      x, y, w, h = cv2.boundingRect(contour)

      # Ignore small contours
      if w > 20 and h > 20:
        # Extract region of interest (ROI)
        roi = gray[y:y + h, x:x + w]

        # Resize ROI to 8x8
        resized_roi = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)

        # Flatten ROI
        flattened_roi = resized_roi.reshape(1, -1)

        # Predict digit using model
        prediction = model.predict(flattened_roi)

        # Display predicted digit on frame
        cv2.putText(frame, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Webcam', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release webcam and destroy windows
  cap.release()
  cv2.destroyAllWindows()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='App para vião computacional.')
    parser.add_argument('-m', '--modelo', type=str, help='Caminho para o arrquivo do modelo.')

    args = parser.parse_args()

    filepath = args.modelo


    with open(filepath, 'rb') as file:
      
      modelo = pickle.load(file)


    predict_digit_from_webcam(modelo)