import matplotlib
matplotlib.use('Agg')
from load_data_inception import transform_data
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
#from torchvision.models.vgg import model_urls
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np
import time

batch_size = 50
use_gpu = torch.cuda.is_available()

def train_new_model(dataloaders, dataset_sizes):
#    global dataloaders, dataset_sizes
    #model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
    
    pre_model = torchvision.models.inception_v3(pretrained=True)
    

    for param in pre_model.parameters():    #----> 1
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    #layers = list(pre_model.children())[:-1]
    #pre_last_layer = list(pre_model.children())[-1]
    #new_last_layer = nn.Linear(4096, 2)
    #layers += [new_last_layer]
    
    new_classifier = nn.Linear(2048,2)
    pre_model.fc = new_classifier
    if use_gpu:
        pre_model = pre_model.cuda()
    criterion = nn.CrossEntropyLoss()
    accuracy = []
    loss_history = []
    learning_rate = [0.001]
    best_acc = 0.0
    best_model = None
    #for param in pre_model.parameters():
    #	print param.requires_grad
    # Observe that all parameters are being optimized
    for lr in learning_rate:
        #filter(lambda p: p.requires_grad, model.parameters())
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, pre_model.parameters()), lr=lr, momentum=0.9,nesterov=True)
        # optimizer = optim.SGD(pre_model.fc.parameters(), lr=lr,momentum=0.9,nesterov=True )
        optimizer = optim.Adam(pre_model.fc.parameters(), lr=lr )
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.00001)

        new_model, val_acc, accuracy, loss_history = train_model(pre_model, criterion, optimizer,
                             exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=100)
        if val_acc > best_acc:
            best_model = new_model
            best_acc = val_acc
        best_model = new_model
    
   # best_model.save_state_dict("best_model.pt")
    torch.save(best_model, "best_model_inception.pt")
    return best_model, accuracy, loss_history


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    i = 0
    accuracy = []
    loss_history = []
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
            best_sens = 0.0
            # Iterate over data.
            rc = 0
            fs = 0
            cm = 0
            for data in dataloaders[phase]:

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
                #print torch.max(outputs[0])
                _, preds = torch.max(outputs[0].data, 1)
                # print preds, labels.data ,"labels"
                loss = sum((criterion(o,labels) for o in outputs))
                #loss1 = criterion(op1, labels)
                #loss2 = criterion(op2,labels)
               # loss = loss1+loss2
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if use_gpu:
                    p = preds.cpu().numpy()
                    l = labels.cpu().data.numpy()
                #loss_epoch.append((i, loss.data[0]))
                # print  confusion_matrix(l,p)
                rec =  recall_score(l,p)
                f1 = f1_score(l,p)
                rc+= rec
                fs += f1
                cm += confusion_matrix(l,p)
                print torch.sum(preds == labels.data), phase
            #print float(tn) / float(tn+fp), "specifity"
            #print float(tp)/float(tp+fn), "sensitivity"
            print float(fs)/(float(dataset_sizes[phase]/ float(batch_size)))
            print float(rc)/(float(dataset_sizes[phase]/float(batch_size))), "sensitivity"
            sens = float(rc)/(float(dataset_sizes[phase]/float(batch_size)))
            if sens > best_sens:
		print "best sens" , best_sens
                best_sens = sens
            epoch_loss = float(running_loss) / float(dataset_sizes[phase])
            epoch_acc = float(running_corrects) / float(dataset_sizes[phase])
            if phase == "train":
                accuracy.append(epoch_acc)
                loss_history.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

           # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print cm
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, best_acc, accuracy, loss_history



def train_model2(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    i = 0
    accuracy = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            
            
            
            # Iterate over data.
            rc = 0 
            fs = 0 
            cm = 0
            for data in dataloaders[phase]:
                
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
                if use_gpu:
                    p = preds.cpu().numpy()
                    l = labels.cpu().data.numpy()
                #loss_epoch.append((i, loss.data[0]))
                # print  confusion_matrix(l,p)
                rec =  recall_score(l,p)
                f1 = f1_score(l,p)
                rc+= rec
                fs += f1
                cm += confusion_matrix(l,p)

                print torch.sum(preds == labels.data), phase
            #print float(tn) / float(tn+fp), "specifity"
            #print float(tp)/float(tp+fn), "sensitivity"
            print float(fs)/(float(dataset_sizes[phase]/ float(batch_size)))
            print float(rc)/(float(dataset_sizes[phase]/float(batch_size))), "sensitivity"
            sens = float(rc)/(float(dataset_sizes[phase]/float(batch_size)))
            if sens > best_sens:
		print "best sensi", sens
            epoch_loss = float(running_loss) / float(dataset_sizes[phase])
            epoch_acc = float(running_corrects) / float(dataset_sizes[phase])
            if phase == "train":
	    	accuracy.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
                
                print cm

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, accuracy

def shifted_to_predict():

    image_datasets, dataloaders = transform_data()
    print image_datasets["train"].classes , "classes"

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
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
    # output = new_model(image_datasets["val"])
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy history')
    fig.savefig('vgg_trained_.png', dpi=fig.dpi)
    plt.show()
