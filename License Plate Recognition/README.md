# License Plate Recognition

Design algorithms to realize license plate recognition, including two traditional visual recognition methods (shape and color) and YOLO method.

## Structure

```
-License Plate Recognition
|---models
	|---ColorLocation.py # traditional color-based method
	|---PlateLocation.py  # traditional shape-based method
	|---YOLO_detection.py # YOLO
	|---LPRNET.py
|---utils
	|---DateLoader.py
	|---utils.py
	|---Others rely on external code
|---main.py 		# Select method (All 3) 
|---params.py       # Set params
|---train.py
```

