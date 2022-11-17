# Code-sample

Here is the code for the four projects.

- **SVHN Digit Classification**

- **Representation Learning**

- **License Plate Recognition**

- **Auto Encoder for Cat**

  In this example, the labeled training data is not enough to directly use BP to train the classifier. However, a good classifier can be trained by using an autoencoder approach.

```
0. Read and prepare data
1. Use unlabeled data (unlabeled set) to train the autoencoder
2. Remove the layers behind the sparse representation layer after training
3. Use the labeled data set (triangular set) to form a new data set in the sparse representation layer
4. Form a new training dataset for the supervised network (encoded training set and its labels)
5. Train the network using the new training dataset
6. Combining two networks
7. Test the network with the test set
```

