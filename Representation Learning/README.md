# Representation Learning
Classification of Occluded Face Images Based on SRC Algorithm and Voting Mechanism

1. Check the data characteristics in the dataset, and determine the image block size. The unoccluded face as the training data, and the occluded face as the test data.
2. Apply the SCR algorithm to build a dictionary and classify the testset based on block voting

## Structure

```
-representation learning
|---dataloader.py  # Load the SVHN dataset
|---main.py 		 
|---params.py       # Set param
|---utils.py
```

## 