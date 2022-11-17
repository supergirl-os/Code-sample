# Digit classification on SVHN

Develop a neural network to implement digit classification on the SVHN streetscape dataset. The neural network uses the simplest fully-connected neural network with BP backpropagation algorithm to update the weights (also using CNN). The loss function and accuracy rate are used to evaluate the classification effect of the model by modifying the number of layers and learning rate of the neural network and using the variable control method.

## Structure

```
-SVHN digit classification
|---model
	|---CNN.py
	|---Network.py  # Fully-connected NN
|---data_loader.py  # Load the SVHN dataset
|---main.py 		 
|---module.py       # Core module for NN
|---utils.py
```

## Requirements

- numpy
- matplotlib
- scipy

