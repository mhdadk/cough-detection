import torch

def train(net,dataloader,loss_func,optimizer,device):
    
    # put in training mode
    
    net.train()
    
    # to compute training accuracy
    
    num_true_pred = 0
    
    # to compute epoch training loss
    
    total_loss = 0
    
    for images,labels in dataloader:
        
        # load onto GPU
        
        images = images.to(device)
        labels = labels.to(device).type_as(images) # needed for BCE loss
        
        # zero the accumulated parameter gradients
        
        optimizer.zero_grad()
        
        # outputs of net for batch input
        
        outputs = net(images)
        
        # compute (mean) loss
        
        loss = loss_func(outputs,labels)
        
        # compute loss gradients with respect to parameters
        
        loss.backward()
        
        # update parameters according to optimizer
        
        optimizer.step()
        
        # record running statistics
        
        # since sigmoid(0) = 0.5, then negative values correspond to class 0
        # and positive values correspond to class 1
        
        class_preds =  outputs > 0 
        num_true_pred = num_true_pred + torch.sum(class_preds == labels)
        
        # since the loss is mean-reduced, loss*images.shape[0] is used to
        # un-average the loss
        
        total_loss = total_loss + (loss*images.shape[0])
    
    train_loss = total_loss.item() / len(dataloader.dataset)
    
    train_acc = num_true_pred.item() / len(dataloader.dataset)
    
    return net,train_loss,train_acc

if __name__ == '__main__':
    
    from AudioDataset import AudioDataset
    
    from torchvision import models
    
    from torch import nn
    
    from torch import optim
    
    import time
    
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
        
    # initialize loss function
    
    num_coughs = sum(train_dataset.labels)
    
    pos_weight = torch.tensor(len(train_dataset)/num_coughs)
    
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # initialize optimizer
    
    optimizer = optim.SGD(params = net.parameters(),
                          lr = 0.00005,
                          momentum = 0.6)
    
    epoch_start = time.time()
    
    net,train_loss,train_acc = train(net,
                                     train_dataloader,
                                     loss_func,
                                     optimizer,
                                     device)
    
    epoch_end = time.time()
    
    epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
    
    print('Training Loss: {:.4f}'.format(train_loss))
    print('Training Accuracy: {:.2f}%'.format(train_acc*100))
    print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        