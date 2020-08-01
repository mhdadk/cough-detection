import torch
from torch import nn
import copy

class Net(nn.Module):
    
    def __init__(self,param_path):
        
        # run nn.Module's constructor
        
        super(Net,self).__init__()
        
        # path to .pkl file containing the pre-trained parameters
        
        self.param_path = param_path
        
        # build the net
        
        self.build_net()
        
        # load the pre-trained parameters
        
        self._load_parameters()
    
    def build_net(self):
        
        """
        
        KEY
        ---------
        
        conv(a,b,c,d):
            a = number of feature maps
            b = b x b kernel size
            c = c x c stride
            d = d x d padding
        batch_norm(a):
            a = number of feature maps
        max_pool(a):
            a = a x a kernel size
        avg_pool():
            Averages all pixel values for each feature map. For example, if
            there are 1024 feature maps, then this returns a 1024-dimensional
            vector.
        
        STAGES
        ---------
        
        Stage 1: conv(16,3,1,1) --> batch_norm(16) --> ReLU --> conv(16,3,1,1)
                 --> batch_norm(16) --> ReLU --> max_pool(2)
        
        Stage 2: conv(32,3,1,1) --> batch_norm(32) --> ReLU --> conv(32,3,1,1)
                 --> batch_norm(32) --> ReLU --> max_pool(2)
        
        Stage 3: conv(64,3,1,1) --> batch_norm(64) --> ReLU --> conv(64,3,1,1)
                 --> batch_norm(64) --> ReLU --> max_pool(2)
        
        Stage 4: conv(128,3,1,1) --> batch_norm(128) --> ReLU --> conv(128,3,1,1)
                 --> batch_norm(128) --> ReLU --> max_pool(2)
        
        Stage 5: conv(256,3,1,1) --> batch_norm(256) --> ReLU --> conv(256,3,1,1)
                 --> batch_norm(256) --> ReLU --> max_pool(2)
        
        Stage 6: conv(512,3,1,1) --> batch_norm(512) --> ReLU --> max_pool(2)
        
        Stage 7: conv(1024,3,1,1) --> batch_norm(1024) --> ReLU
        
        NETWORK ARCHITECTURE
        ---------------------
        
        Stage 1 --> Stage 2 --> Stage 3 --> Stage 4 --> Stage 5 --> Stage 6
        --> Stage 7 --> avg_pool()
        
        """
        
        in_channels = 16
        
        conv1 = nn.Conv2d(in_channels = 1,
                          out_channels = in_channels,
                          kernel_size = (3,3),
                          padding = (1,1))
        
        """
        See here for details on why two different batch_norm (or any trainable) 
        layers need to be used:
            
        https://discuss.pytorch.org/t/modifying-the-state-dict-changes-the-values-of-the-parameters/89030
        
        """
        
        batch_norm1 = nn.BatchNorm2d(num_features = 16)
        
        activation = nn.ReLU()
        
        conv2 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels,
                          kernel_size = (3,3),
                          padding = (1,1))
        
        batch_norm2 = nn.BatchNorm2d(num_features = 16)
        
        pooling = nn.MaxPool2d(kernel_size = (2,2))
        
        # first stage
        
        stages = [nn.Sequential(conv1,
                                batch_norm1,
                                activation,
                                conv2,
                                batch_norm2,
                                activation,
                                pooling)]
        
        # next 4 stages
        
        for i in range(4):
            
            conv1 = nn.Conv2d(in_channels = in_channels,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            batch_norm1 = nn.BatchNorm2d(num_features = in_channels * 2)
            
            conv2 = nn.Conv2d(in_channels = in_channels * 2,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            batch_norm2 = nn.BatchNorm2d(num_features = in_channels * 2)
            
            stages += [nn.Sequential(conv1,
                                     batch_norm1,
                                     activation,
                                     conv2,
                                     batch_norm2,
                                     activation,
                                     pooling)]
            
            in_channels = in_channels * 2
            
        # 6th stage, in_channels = 256
        
        conv1 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels * 2,
                          kernel_size = (3,3),
                          padding = (1,1))
            
        batch_norm = nn.BatchNorm2d(num_features = in_channels * 2)
            
        stages += [nn.Sequential(conv1,
                                 batch_norm,
                                 activation,
                                 pooling)]
        
        in_channels = in_channels * 2
        
        # final stage, in_channels = 512
        
        conv1 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels * 2,
                          kernel_size = (2,2),
                          padding = (0,0))
            
        batch_norm = nn.BatchNorm2d(num_features = in_channels * 2)
            
        stages += [nn.Sequential(conv1,
                                 batch_norm,
                                 activation)]
        
        # assign names to the stages for the state_dict
        
        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]
        self.stage5 = stages[4]
        self.stage6 = stages[5]
        self.stage7 = stages[6]
    
    def _load_parameters(self):
        
        """
        Assign parameters from .pkl file to layers.
        """
        
        # load the state dict from the .pkl file
        
        self.old_state_dict = torch.load(f = self.param_path,
                                         map_location = torch.device('cpu'))
        
        # make a copy to use load_state_dict() method later. deepcopy needed
        # because dict is a mutable object
        
        state_dict = copy.deepcopy(self.state_dict())
        
        for key,value in self.old_state_dict.items():
            
            # skip layer 19
            
            if key[12:14].isdigit() and int(key[12:14]) == 19:
                continue
            
            # get the name of the parameter in the new state dict corresponding
            # to the name of the parameter in the old state dict
            
            parameter_name = self.map_param_name(key)
            
            state_dict[parameter_name] = value
        
        # modify the net's state dict
        
        self.load_state_dict(state_dict)
        
    def map_param_name(self,key):
        
        """
        Maps names of parameters in the old state dict to names of parameters
        in the new state dict.
        """
        
        # get layer number in the old state dict
        
        if key[12:14].isdigit():
            layer_num = int(key[12:14])
        else:
            layer_num = int(key[12])
        
        # get sub-layer number in the old state dict. 0 means conv layer and
        # 1 means batch norm layer
        
        sub_layer_num = int(key.split('.')[2])
        
        # get parameter type in the old state dict. This can be 'weight' or
        # 'running_mean', for example
        
        parameter_type = key.split('.')[-1]
        
        # map layer number and sub-layer number in the old state dict to stage
        # number and sub-stage number in the new state dict. Note that only
        # convolutional layers and batch normalization layers have parameters
        
        if layer_num == 1 or layer_num == 2:
            stage = '1'
        elif layer_num == 4 or layer_num == 5:
            stage = '2'
        elif layer_num == 7 or layer_num == 8:
            stage = '3'
        elif layer_num == 10 or layer_num == 11:
            stage = '4'
        elif layer_num == 13 or layer_num == 14:
            stage = '5'
        elif layer_num == 16:
            stage = '6'
        elif layer_num == 18:
            stage = '7'
        
        sub_stage = self._get_sub_stage_number(layer_num,sub_layer_num)
        
        parameter_name = 'stage'+stage+'.'+sub_stage+'.'+parameter_type
        
        return parameter_name
        
    def _get_sub_stage_number(self,layer_num,sub_layer_num):
        
        """
        Helper function to return the sub-stage number in the new state dict
        """
        
        if layer_num < 16:
        
            # if conv and first layer
            
            if sub_layer_num == 0 and ((layer_num % 3) % 2) == 1: 
                sub_stage_num = '0'
            
            # if batch norm and first layer
            
            elif sub_layer_num == 1 and ((layer_num % 3) % 2) == 1:
                sub_stage_num = '1'
                
            # if conv and second layer
            
            elif sub_layer_num == 0 and ((layer_num % 3) % 2) == 0: 
                sub_stage_num = '3'
            
            # if batch norm and second layer
            
            else: 
                sub_stage_num = '4'
        
        else:
            
            # if conv layer
            
            if sub_layer_num == 0:
                sub_stage_num = '0'
            
            # if batch norm layer
            
            else:
                sub_stage_num = '1'
        
        return sub_stage_num
    
    def forward(self,x):
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        # average of all pixels in each feature map
        
        x = nn.functional.avg_pool2d(input = x,
                                     kernel_size = x.shape[2:])
        
        # flatten from N x 1024 x 1 x 1 to N x 1024, where N is the batch size
        
        x = torch.flatten(input = x,
                          start_dim = 1)
        
        return x

# check that the two state dicts are equal

if __name__ == '__main__':
    
    param_path = 'mx-h64-1024_0d3-1.17.pkl'
    
    net = Net(param_path)
    
    for key,value in net.old_state_dict.items():
        
        # skip layer 19
        
        if '19' in key:
            continue
        
        param1 = value
        param2 = net.state_dict()[net.map_param_name(key)]
        
        # torch.allclose() because parameters are floats
        
        is_equal = torch.allclose(param1,param2)
        
        print(is_equal)
