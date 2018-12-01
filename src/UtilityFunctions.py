import numpy as np;
from src import DeepNeuralNetworkStarter as DNNS;
import matplotlib.pyplot as plt;

desent_optimzation_map = {};
desent_optimzation_map[0] = "No_momentum";
desent_optimzation_map[1] = "Poly";
desent_optimzation_map[2] = "Nestrov";
desent_optimzation_map[3] = "Adam";
desent_optimzation_map[4] = "RMSProp";


def accuracy(Y_pred, Y_label):
    Y = np.argmax(Y_pred, axis=0)
    correct = 0
    Y_label = Y_label.flatten()
    for i in range(Y_label.size):
        if Y[i] == Y_label[i]:
            correct += 1
    return correct / Y_label.size * 100


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b);
    p = np.random.permutation(len(a));
    return a[p].T, b[p].T;


def getTrainAndValidationAccuracy(train_data_act, train_label_act, validation_data, validation_label, test_data,
                                  test_act, parameters):
    # compute the accuracy for training set, validation set and testing set by predicting them first
    trPred, train_loss = DNNS.classify(train_data_act, parameters, train_label_act);
    vdPred, validation_loss = DNNS.classify(validation_data, parameters, validation_label);
    tsPred, test_loss = DNNS.classify(test_data, parameters, test_act);
    trAcc = accuracy(trPred, train_label_act);
    valAcc = accuracy(vdPred, validation_label);
    tsAcc = accuracy(tsPred, test_act);
    print("Accuracy for training set is {0:0.3f} %".format(trAcc));
    print("Accuracy for validation set is {0:0.3f} %".format(valAcc));
    print("Accuracy for test set is {0:0.3f} %".format(tsAcc));


def plotWithCosts(num_iterations, costList, is_batch_comparision=True, net_dims=[], batch_size=512, total_size=10240,
                  is_learning_comparision=False, learning_rate=0, OT=0):
    # PLOT of costs vs iterations
    # here plot our results where our x axis would be the 1 to no. of iteration with interval of 10
    # y axis would be costs list for training and validation set
    colorlist = ["blue", "green", "red", "yellow", "magenta", "black", "orange", "cyan"]
    presionSize = 1 / (total_size / batch_size);
    iterations = [i for i in np.arange(0, num_iterations, presionSize)];
    c = 0;
    title = "Training loss for %s dimensions multi layer neurons" % str(net_dims[:len(net_dims) - 1]);
    if is_batch_comparision:
        title += " \n With Learaning rate  " + str(learning_rate);
        title += " \n and Optimatization Technique   " + desent_optimzation_map[OT];
    else:
        if is_learning_comparision:
            title += " \n  With Batch Size  " + str(batch_size);
            title += " \n  and Optimatization Technique   " + desent_optimzation_map[OT];
        else:
            title += " \n  With Learaning rate  " + str(learning_rate);
            title += " \n  With Batch Size  " + str(batch_size);
    for key in costList:
        label = "";
        if is_batch_comparision:
            label = "Batch size of " + str(key);
            presionSize = 1 / (total_size / key);
            iterations = [i for i in np.arange(0, num_iterations, presionSize)];
        else:
            if is_learning_comparision:
                label = "learning rate of " + str(key);
            else:
                label = desent_optimzation_map[key];
        plt.plot(iterations, costList[key], label=label, color=colorlist[c % len(colorlist)]);
        c += 1
    plt.legend();
    plt.title(title);
    plt.show();
