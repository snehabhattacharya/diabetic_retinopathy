import matplotlib
matplotlib.use('Agg')
from load_data import transform_data
from sklearn.metrics import confusion_matrix, recall_score
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np
import time


use_gpu = torch.cuda.is_available()

def train_new_model():
    model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
    
    pre_model = torchvision.models.vgg19_bn(pretrained=True)
    for param in pre_model.parameters():    #----> 1
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    layers = list(pre_model.classifier.children())[:-1]
    pre_last_layer = list(pre_model.classifier.children())[-1]
    new_last_layer = nn.Linear(pre_last_layer.in_features, 2)
    layers += [new_last_layer]
    
    new_classifier = nn.Sequential(*layers)
    pre_model.classifier = new_classifier

    criterion = nn.CrossEntropyLoss()

    learning_rate = [0.001]
    best_acc = 0.0
    best_model = None
    # Observe that all parameters are being optimized
    for lr in learning_rate:
        optimizer = optim.SGD(pre_model.classifier[6].parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        new_model, val_acc , loss_history = train_model(pre_model, criterion, optimizer,
                             exp_lr_scheduler, num_epochs=2)
        if val_acc > best_acc:
            best_model = new_model
            best_acc = val_acc

    return best_model, loss_history

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    i = 0
    loss_epoch = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            
            
            
            # Iterate over data.
            for data in dataloaders[phase]:
                rc = 0
                print len(dataloaders["train"])
                # get the inputs
                inputs, labels = data
                
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                print inputs.size()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # print preds, labels.data ,"labels"
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                p = preds.numpy()
                l = labels.data.numpy()
                loss_epoch.append((i, loss.data[0]))
                print  confusion_matrix(l,p)
                rc =  recall_score(l,p)
                #fp += cm[0][1]
                #fn += cm[1][0]
                #tp += cm[1][1]
                print torch.sum(preds == labels.data), phase
            #print float(tn) / float(tn+fp), "specifity"
            #print float(tp)/float(tp+fn), "sensitivity"
            print float(rc)/float(dataset_sizes[phase]), "sensitivity"
            epoch_loss = float(running_loss) / float(dataset_sizes[phase])
            epoch_acc = float(running_corrects) / float(dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, loss_epoch

if __name__ == "__main__":

    image_datasets, dataloaders = transform_data()
    print image_datasets["train"].classes , "classes"

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    # print image_datasets['train'][0]
    # get some random training images
    # train_loader, val_loader = transform_data_deprecate()
    # dataloaders = {}
    # dataloaders['train'] = train_loader
    # dataloaders['val'] = val_loader

    # dataset_sizes = {}
    # dataset_sizes['train'] = 6
    # dataset_sizes['val'] = 4

    # print "dataset = ", dataset_sizes
    fig = plt.figure()
    
    new_model, loss_history = train_new_model()
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    fig.savefig('temp.png', dpi=fig.dpi)
    plt.show()
