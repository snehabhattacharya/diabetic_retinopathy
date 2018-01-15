import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model_inception_v3 import train_new_model
from load_data_inception import transform_data
from torch.autograd import Variable
from sklearn.metrics import recall_score
use_gpu = torch.cuda.is_available()

def predict_dr(model, dataloader, dataset_size):
    i = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0.0
    rec = 0
    for data in dataloaders['val']:
	# get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        if use_gpu:
	    inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        print inputs.size()
        outputs = model(inputs)
        print outputs
        _, preds = torch.max(outputs[0].data, 1)
        loss=sum((criterion(o,labels) for o in outputs))
        rc = recall_score(preds.cpu().numpy(), labels.cpu().data.numpy())
        # statistics
        rec += rc
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
    total_sensitivy = float(rec)/(float(dataset_size)/float(50))
    total_loss = float(running_loss) / dataset_size
    acc = float(running_corrects) / dataset_size

    print('Loss: {:.4f} Acc: {:.4f}'.format(
        total_loss, acc))
    print total_sensitivy, "sens"


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
        loss_file = open('loss_inc.txt', 'w')
        for loss in loss_history:
            loss_file.write("%s\n" % loss)
        loss_file.close()

        acc_file = open('acc_inc.txt', 'w')
        for acc in accuracy:
            acc_file.write("%s\n" % acc)
        acc_file.close()

        fig = plt.figure()
        plt.plot(accuracy)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Training accuracy history')
        fig.savefig('inception_trained_.png', dpi=fig.dpi)
        plt.show()
    
    else:
        model = torch.load('best_model_inception.pt')
   
    predict_dr(model, dataloaders, dataset_sizes['val'])

