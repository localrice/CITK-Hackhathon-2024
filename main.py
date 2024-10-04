import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

car = "car3.jpeg"  # Path to your car image
plt.figure(figsize=(10,10))

# Load the image
img = cv.imread(car)

# Original image
plt.subplot(421)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.xticks([]), plt.yticks([])

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(422)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# Apply bilateral filter for noise reduction
bfilter = cv.bilateralFilter(gray, 11, 17, 17)
plt.subplot(423)
plt.imshow(bfilter, cmap='gray')
plt.title('Processed (Noise Reduction)')
plt.xticks([]), plt.yticks([])

# Edge detection using Canny
edged = cv.Canny(bfilter, 30, 200)
plt.subplot(424)
plt.imshow(edged, cmap='gray')
plt.title('Edge Detection')
plt.xticks([]), plt.yticks([])

# Find contours in the edge-detected image
keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in sorted_contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is None:
    print("No plate-like contour detected.")
else:
    # Create a mask and extract the detected license plate area
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    new_image = cv.bitwise_and(img, img, mask=mask)
    plt.subplot(425)
    plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
    plt.title('Masked Image')
    plt.xticks([]), plt.yticks([])

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    plt.subplot(426)
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Image')
    plt.xticks([]), plt.yticks([])

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Filter out small bounding boxes and irrelevant texts like 'IND'
    valid_texts = []
    for (bbox, text, _) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox

        # Get width and height of the bounding box
        bbox_width = abs(top_right[0] - top_left[0])
        bbox_height = abs(bottom_left[1] - top_left[1])

        # Exclude 'IND' and any bounding boxes that are too small
        if text != "IND" and bbox_width > 50 and bbox_height > 20:
            valid_texts.append(text)

    # If valid text is detected, use the first one (most confident)
    if valid_texts:
        text = valid_texts[0]
    else:
        text = "No Text Found"
    print(text)
    # Display the detected text on the original image
    font = cv.FONT_HERSHEY_SIMPLEX
    res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), 
                     fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    
    # Show the final result with the text overlay
    plt.subplot(427)
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.title('Final Image with Text')
    plt.xticks([]), plt.yticks([])

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()
