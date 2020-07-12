import torch

from torch import nn

from torch import optim

from torchvision import models

import copy

from AudioDataset import AudioDataset

import time

from train import train

from validate import validate

# to put tensors on GPU if available
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get the pretrained model

net = models.mobilenet_v2(pretrained=True,progress=True)

# modify last layer to account for two classes

in_features = net.classifier[-1].in_features

out_features = 1

net.classifier[-1] = nn.Linear(in_features = in_features,
                               out_features = out_features,
                               bias = True)

# initialize datasets and dataloaders

data_dir = '../../data'

sample_rate = 22050

train_dataset = AudioDataset(data_dir,sample_rate,'train')

train_dataloader = torch.utils.data.DataLoader(
                   dataset = train_dataset,
                   batch_size = 8,
                   shuffle = True,
                   num_workers = 0,
                   pin_memory = (device == 'cuda'))

val_dataset = AudioDataset(data_dir,sample_rate,'val')

val_dataloader = torch.utils.data.DataLoader(
                 dataset = val_dataset,
                 batch_size = 8,
                 shuffle = True,
                 num_workers = 0,
                 pin_memory = (device == 'cuda'))

test_dataset = AudioDataset(data_dir,sample_rate,'test')

test_dataloader = torch.utils.data.DataLoader(
                  dataset = test_dataset,
                  batch_size = 8,
                  shuffle = True,
                  num_workers = 0,
                  pin_memory = (device == 'cuda'))

# initialize loss function

num_coughs = sum(train_dataset.labels)

pos_weight = torch.tensor(len(train_dataset)/num_coughs)

loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# initialize optimizer

optimizer = optim.SGD(params = net.parameters(),
                      lr = 0.00005,
                      momentum = 0.4)

# initialize learning rate scheduler
    
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer,
                                      step_size = 3,
                                      gamma = 0.5,
                                      last_epoch = -1)

# number of epochs to train and validate for

num_epochs = 20

# best validation accuracy

best_val_acc = 0

# starting time

start = time.time()

for epoch in range(num_epochs):
    
    epoch_start = time.time()
    
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 30)
    
    net,train_loss,train_acc = train(net,
                                     train_dataloader,
                                     loss_func,
                                     optimizer,
                                     device)
    
    print('Training Loss: {:.4f}'.format(train_loss))
    print('Training Accuracy: {:.2f}%'.format(train_acc*100))
    
    val_loss,val_acc = validate(net,
                                val_dataloader,
                                loss_func,
                                device)   
    
    scheduler.step()
    
    print('Validation Loss: {:.4f}'.format(train_loss))
    print('Validation Accuracy: {:.2f}%'.format(train_acc*100))
    
    epoch_end = time.time()
    
    epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
    
    print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
    
    # save the weights for the best validation accuracy
        
    if val_acc > best_val_acc:
        
        print('Saving checkpoint...')
        
        best_val_acc = val_acc
        
        # deepcopy needed because a dict is a mutable object
        
        best_parameters = copy.deepcopy(net.state_dict())
        
        torch.save(net.state_dict(),
                   '../../models/mobilenet_best.pt')

end = time.time()
total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
print('\nTotal Time Elapsed (HH:MM:SS): ' + total_time)
print('Best Validation Accuracy: {:.2f}%'.format(best_val_acc*100))