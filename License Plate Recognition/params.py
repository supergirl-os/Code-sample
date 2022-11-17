# Enable debug module?
debug = True

# === For PlateLocation-> locate ===
# Variables for Gaussian Blur
GaussianBlurSize = (5, 5)  # can select 3,7,0 to see the performance
# The minimum value of the contour area to filter out too small contours, Unit: px
MIN_AREA = 50

# === For PlateLocation-> judge ===
Height = 14.0  # Chinese license plate height is about 14cm
Width = 44.0  # Chinese license plate length is about 44cm
Aspect = 3.2  # License plate aspect ratio, Spanish standard license plate aspect ratio is 4.7272,
                # Chinese license plate aspect ratio is 3.142857
Error = 0.3  # License plate aspect ratio error
StandardArea = Height * Width  # Standard area

VerifyMin = 1.5  # Scale factor for minimum area
VerifyMax = 120  # scaling factor for the largest area

# === For ColorLocation ===
# ColorMin = [70, 160, 130]  # Set bounds (HSV)
# ColorMax = [130, 255,255]
ColorMin = [50, 60, 56]  # Set bounds (HSV)
ColorMax = [124, 255, 255]
blur = False  # blur?

# === For YOLO_detection ===
Recognize = True
