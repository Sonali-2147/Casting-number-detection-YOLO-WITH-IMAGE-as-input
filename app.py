#Importing required libraries
import cv2
from ultralytics import YOLO
import pytesseract

# Load YOLOv8 model 
model = YOLO('yolov5n.pt')

# extract detected regions and recognize text
def extract_and_recognize_numbers(image, results, box_reduction_factor=0.1, min_difference=5):
    output_images = [] 
    bounding_boxes = [] 
    recognized_numbers = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0] 

            # If confidence is above a threshold, process the detected region
            if confidence > 0.3:  
               
                width = x2 - x1
                height = y2 - y1
                x1 = int(x1 + box_reduction_factor * width)
                y1 = int(y1 + box_reduction_factor * height)
                x2 = int(x2 - box_reduction_factor * width)
                y2 = int(y2 - box_reduction_factor * height)

                detected_region = image[y1:y2, x1:x2] 
                # Recognize text (number) 
                text = pytesseract.image_to_string(detected_region, config='--psm 6')  
                
                # Extract digits
                filtered_text = ''.join(filter(str.isdigit, text.strip()))

                if filtered_text:
                    
                    number = int(filtered_text)
                    is_similar = False
                    
                    # Check if the number is similar to any previously detected number
                    for previous_number, _ in recognized_numbers:
                        if abs(number - previous_number) < min_difference:
                            is_similar = True
                            break
                    
                    if not is_similar:
                        recognized_numbers.append((number, (x1, y1, x2, y2, filtered_text)))
                        output_images.append(detected_region)  
                        bounding_boxes.append((x1, y1, x2, y2, filtered_text)) 

    return output_images, bounding_boxes

# Load input image
image_path = 'input.png'  
image = cv2.imread(image_path)

# Resize the image to match the expected input size for the YOLO model
image_resized = cv2.resize(image, (640, 640)) 


results = model(image_resized)

# Extract regions containing numbers and recognize text
detected_regions, bounding_boxes = extract_and_recognize_numbers(image, results, box_reduction_factor=0.1, min_difference=4)

# Draw bounding boxes around the detected regions
for (x1, y1, x2, y2, text) in bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(image, f"Detected Numbers: {text}", (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Detected Numbers and Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image with bounding boxes
cv2.imwrite("output_image_with_boxes.png", image)

