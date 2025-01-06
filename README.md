# Casting-number-detection-YOLO-WITH-IMAGE-as-input



Here is a README file for the provided code:

---

# YOLOv8 + Tesseract for Number Detection and Text Recognition

This Python project uses YOLOv8 for detecting regions containing numbers in an image and Tesseract OCR to extract and recognize those numbers. The detected regions are processed and bounding boxes with the recognized text are drawn on the output image.

## Requirements

1. **Python 3.7+**
2. **Libraries:**
   - OpenCV
   - Ultralytics YOLOv8
   - Tesseract
   - pytesseract
   - NumPy

You can install the required libraries using the following commands:

```bash
pip install opencv-python-headless ultralytics pytesseract numpy
```

Additionally, you need to install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system.

## Functionality

1. **YOLOv8 Model**: The model (`yolov8n.pt`) is used for detecting objects in an image.
2. **Text Recognition**: Once the objects are detected, the program extracts regions of interest (bounding boxes around detected objects) and uses Tesseract OCR to extract and recognize text (numbers in this case) from these regions.
3. **Bounding Box**: Bounding boxes are drawn on the image around the detected regions with the corresponding recognized text.
4. **Text Filtering**: Only digits are extracted from the recognized text, and they are added to the output if they are unique (based on a minimum difference threshold).

## How to Run

1. Clone or download the project repository.
2. Place an input image (e.g., `input.png`) in the project directory.
3. Run the Python script to perform object detection, text extraction, and display the results:

```bash
python detect_and_recognize.py
```

## Code Breakdown

### Importing Libraries
```python
import cv2
from ultralytics import YOLO
import pytesseract
```
This imports the necessary libraries for image processing, YOLO object detection, and text recognition using Tesseract OCR.

### Load YOLOv8 Model
```python
model = YOLO('yolov8n.pt')
```
The pre-trained YOLOv8 model is loaded.

### Function to Extract and Recognize Numbers
```python
def extract_and_recognize_numbers(image, results, box_reduction_factor=0.1, min_difference=5):
```
This function processes the detected regions from the YOLO model, extracts text using Tesseract OCR, and filters out any duplicates by comparing the recognized numbers.

### Image Processing and Recognition
The image is read and resized to match the input size expected by the YOLO model. The detection results are passed to the function `extract_and_recognize_numbers` to extract the detected regions and their corresponding text.

### Drawing Bounding Boxes
```python
for (x1, y1, x2, y2, text) in bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Detected Numbers: {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```
Bounding boxes are drawn around the detected numbers, and the recognized text is displayed above each box.

### Display and Save Results
```python
cv2.imshow("Detected Numbers and Text", image)
cv2.imwrite("output_image_with_boxes.png", image)
```
The processed image is displayed, and the output with bounding boxes is saved as `output_image_with_boxes.png`.

## Example Output

The output will be an image showing bounding boxes around detected regions with the recognized text (numbers) displayed above each box.

---

## Contact
For any queries or feedback, please email me at kadamsonali2147@gmail.com.

