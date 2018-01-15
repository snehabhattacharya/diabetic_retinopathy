import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import train_new_model
from load_data import transform_data
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import itertools
from itertools import cycle
from sklearn.metrics import confusion_matrix, f1_score

use_gpu = torch.cuda.is_available()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.png')
    plt.close()

def predict_dr(model, dataloader, dataset_size, image_datasets):
    i = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0.0
    total_preds = []
    total_labels = []
    for data in dataloaders['val']:
	# get the inputs
        inputs, labels = data
        total_labels.append(labels.cpu().numpy())
        # wrap them in Variable
        if use_gpu:
	    inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        print inputs.size()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

        # for roc
        total_preds.append(preds.cpu().numpy())
        #total_labels += labels

    total_loss = float(running_loss) / dataset_size
    acc = float(running_corrects) / dataset_size

    print('Loss: {:.4f} Acc: {:.4f}'.format(
        total_loss, acc))

    # roc
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    total_preds = label_binarize(total_preds, classes=[0,1])
    total_labels = label_binarize(total_labels, classes=[0,1])

    confusion = np.array([[0, 0], [0, 0]])
    confusion = confusion_matrix(total_labels, total_preds)
    print confusion
    score = f1_score(total_labels, total_preds)
    print score

    np.set_printoptions(precision=2)
    class_names = image_datasets['val'].classes
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion, classes=class_names,
                      title='Confusion matrix')








    #print total_preds
    #print total_labels
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    fpr[0], tpr[0], _ = roc_curve(total_preds, total_labels)
    print fpr
    roc_auc[0] = auc(fpr[0], tpr[0])

#    for i in range(n_classes):
 #       fpr[i], tpr[i], _ = roc_curve(total_preds[:, i], total_labels[:, i])
  #      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(total_preds.ravel(), total_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #fpr_file = open('fpr_vgg19.txt', 'w')
    #for loss in fpr:
    #    loss_file.write("%s\n" % loss)
    #loss_file.close()
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes-1)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes-1):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes-1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
#    plt.show()
    plt.savefig('./roc.png')
    plt.close()

if __name__ == "__main__":

    image_datasets, dataloaders = transform_data()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    
    # get some random training images
    # train_loader, val_loader = transform_data_deprecate()
    # dataloaders = {}
    # dataloaders['train'] = train_loader
    # dataloaders['val'] = val_loader

    # dataset_sizes = {}
    # dataset_sizes['train'] = 6
    # dataset_sizes['val'] = 4

    # print "dataset = ", dataset_sizes

    train = False
    loss_history = 0.0
    if train:
        model, accuracy, loss_history = train_new_model(dataloaders, dataset_sizes)
            # print "dataset = ", dataset_sizes
        loss_file = open('loss.txt', 'w')
        for loss in loss_history:
            loss_file.write("%s\n" % loss)
        loss_file.close()

        acc_file = open('acc.txt', 'w')
        for acc in accuracy:
            acc_file.write("%s\n" % acc)
        acc_file.close()

        fig = plt.figure()
        plt.plot(accuracy)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Training accuracy history')
        fig.savefig('vgg_trained_.png', dpi=fig.dpi)
        plt.show()
    
    else:
        model = torch.load('best_model_vgg19.pt')
   
    predict_dr(model, dataloaders, dataset_sizes['val'], image_datasets)

