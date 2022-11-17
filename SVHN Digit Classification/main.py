# ==================================================================
# Course: Understanding Deep Neural Networks
# Teacher: Zhang Yi
# Student: Wang Yaxuan
# ID:   2019141440341
#
# Ten-category classification problem on SVHN dataset
# ====================================================================
import time
from data_loader import data_loader, loader_for_conv
from model.Network import Network
from model.CNN import *


# main
def main(model_name,L,alpha,epochs,batch_size,if_scheduler,optimizer):
    if model_name == "Network":
        print("Start read data")
        time_1 = time.time()
        x_train, train_labels, x_test, test_labels = data_loader()
        net = Network(L, alpha, epochs, batch_size, if_scheduler)
        net.designer()
        time_2 = time.time()
        print("read data cost ", time_2 - time_1, ' second', '\n')
        # model training
        print('Start training')
        net.train(x_train, train_labels, x_test, test_labels)
        time_3 = time.time()
        print("training and predicting cost ", time_3 - time_2, ' second', '\n')
        net.show()
        net.save_model()
    elif model_name == "CNN":
        x_train, train_labels, x_test, test_labels = loader_for_conv()
        net = ConvNet(input_dim=(3,32,32),
                      conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                      hidden_size=100, output_size=10, weight_init_std=0.01)
        t = Trainer(batch_size, epochs, net, optimizer=optimizer)
        t.train(x_train, train_labels, x_test, test_labels)
        net.save_params("params.pkl")
        print("Saved Network Parameters!")
    elif model_name == "Transformer":
        pass
    else:
        print("Error of model_name")
        return 0


if __name__ == '__main__':
    # param setting
    # model = "Network"  # model type (Network,CNN)
    model = "CNN"
    # Network
    L = 6                   # Network layer (3，6，8)
    if_scheduler = False    # lr schedule on?
    alpha = 0.5             # learning rate
    # CNN
    optimizer = "SGD"       # SGD，Adam
    # common parameters
    epochs = 50             # Iterations
    batch_size = 100        # 

    main(model, L, alpha, epochs, batch_size, if_scheduler,optimizer)


