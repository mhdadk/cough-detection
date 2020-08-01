import torch

from sklearn.metrics import confusion_matrix

def test(net,dataloader,device):
    
    # put in testing mode
    
    net.eval()
    
    # store class predictions
    
    class_preds = []
    
    labels_all = []
    
    for images,labels in dataloader:
        
        # load onto GPU
        
        images = images.to(device)
        
        # store labels
        
        labels_all.extend(labels.tolist())
        
        # don't compute grad_fn to conserve RAM
        
        with torch.set_grad_enabled(False):
        
            # outputs of net for batch input
            
            outputs = net(images)
        
        # since sigmoid(0) = 0.5, then negative values correspond to class 0
        # and positive values correspond to class 1
        
        class_preds.extend((outputs > 0).squeeze().tolist())
    
    CM = confusion_matrix(labels_all,class_preds,labels=[0,1])
    
    TP = CM[1,1]
    
    TN = CM[0,0]
    
    FP = CM[0,1]
    
    FN = CM[1,0]
    
    sensitivity = TP/(TP+FN) # true positive rate (TPR)
    
    specificity = TN/(TN+FP) # true negative rate (TNR)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    balanced_accuracy = (sensitivity+specificity)/2
    
    # Matthews correlation coefficient
    
    MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
    
    # positive predictive value (or precision)
    
    PPV = TP/(TP+FP)
    
    # negative predictive value
    
    NPV = TN/(TN+FN)
    
    metrics = {
        
        'CM':CM,
        'sens':sensitivity,
        'spec':specificity,
        'acc':accuracy,
        'bal_acc':balanced_accuracy,
        'MCC':MCC,
        'PPV':PPV,
        'NPV':NPV
        
    }
    
    return metrics

if __name__ == '__main__':
    
    from AudioDataset import AudioDataset
    
    from torchvision import models
    
    from torch import nn
        
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
    
    # load best parameters
    
    net.load_state_dict(torch.load('../../models/mobilenet_best.pt'))
    
    # initialize datasets and dataloaders
    
    data_dir = '../../data'
    
    sample_rate = 22050
    
    test_dataset = AudioDataset(data_dir,sample_rate,'test')
    
    test_dataloader = torch.utils.data.DataLoader(
                       dataset = test_dataset,
                       batch_size = 8,
                       shuffle = True,
                       num_workers = 0,
                       pin_memory = (device == 'cuda'))
    
    metrics = test(net,test_dataloader,device)
    
    print('\nConfusion Matrix:\n{}\n'.format(metrics['CM']))
    print('Sensitivity/Recall: {:.3f}'.format(metrics['sens']))
    print('Specificity: {:.3f}'.format(metrics['spec']))
    print('Accuracy: {:.3f}'.format(metrics['acc']))
    print('Balanced Accuracy: {:.3f}'.format(metrics['bal_acc']))
    print('Matthews correlation coefficient: {:.3f}'.format(metrics['MCC']))
    print('Precision/PPV: {:.3f}'.format(metrics['PPV']))
    print('NPV: {:.3f}'.format(metrics['NPV']))
        