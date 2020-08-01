import torch

def validate(net,dataloader,loss_func,device):
    
    # put in testing mode
    
    net.eval()
    
    # to compute validation accuracy
    
    num_true_pred = 0
    
    # to compute epoch validation loss
    
    total_loss = 0
    
    for images,labels in dataloader:
        
        # load onto GPU
        
        images = images.to(device)
        labels = labels.to(device).type_as(images) # needed for BCE loss
                
        # don't compute gradients to conserve RAM
        
        with torch.set_grad_enabled(False):
        
            # outputs of net for batch input
            
            outputs = net(images)
            
            # compute (mean) loss
            
            loss = loss_func(outputs,labels)
            
            # since sigmoid(0) = 0.5, then negative values correspond to class 0
            # and positive values correspond to class 1
            
            class_preds =  outputs > 0 
        
        # record running statistics
        
        num_true_pred = num_true_pred + torch.sum(class_preds == labels)
        
        # since the loss is mean-reduced, loss*images.shape[0] is used to
        # un-average the loss
        
        total_loss = total_loss + (loss*images.shape[0])
    
    val_loss = total_loss.item() / len(dataloader.dataset)
    
    val_acc = num_true_pred.item() / len(dataloader.dataset)
    
    return val_loss,val_acc

if __name__ == '__main__':
    
    from AudioDataset import AudioDataset
    
    from torchvision import models
    
    from torch import nn
    
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
    
    val_dataset = AudioDataset(data_dir,sample_rate,'val')
    
    val_dataloader = torch.utils.data.DataLoader(
                       dataset = val_dataset,
                       batch_size = 8,
                       shuffle = True,
                       num_workers = 0,
                       pin_memory = (device == 'cuda'))
        
    # initialize loss function
    
    num_coughs = sum(val_dataset.labels)
    
    pos_weight = torch.tensor(len(val_dataset)/num_coughs)
    
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    epoch_start = time.time()
    
    val_loss,val_acc = validate(net,
                                val_dataloader,
                                loss_func,
                                device)
    
    epoch_end = time.time()
    
    epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
    
    print('Validation Loss: {:.4f}'.format(val_loss))
    print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
    print('Epoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        