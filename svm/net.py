import torch
from torch import nn

class Net(nn.Module):
    
    def __init__(self,param_path):
        
        # run nn.Module's constructor
        
        super(Net,self).__init__()
        
        # save param_path property
        
        self.param_path = param_path
        
        # build net
        
        in_channels = 16
        
        conv1 = nn.Conv2d(in_channels = 1,
                         out_channels = in_channels,
                         kernel_size = (3,3),
                         padding = (1,1))
        
        batch_norm = nn.BatchNorm2d(num_features = 16)
        
        activation = nn.ReLU()
        
        conv2 = nn.Conv2d(in_channels = in_channels,
                         out_channels = in_channels,
                         kernel_size = (3,3),
                         padding = (1,1))
        
        pooling = nn.MaxPool2d(kernel_size = (2,2))
        
        # first stage
        
        stages = [nn.Sequential(conv1,
                                batch_norm,
                                activation,
                                conv2,
                                batch_norm,
                                activation,
                                pooling)]
        
        # next 4 stages
        
        for i in range(4):
            
            conv1 = nn.Conv2d(in_channels = in_channels,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            batch_norm = nn.BatchNorm2d(num_features = in_channels * 2)
            
            conv2 = nn.Conv2d(in_channels = in_channels * 2,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            stages += [nn.Sequential(conv1,
                                     batch_norm,
                                     activation,
                                     conv2,
                                     batch_norm,
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
        
        # load the pre-trained parameters
        
        self._load_parameters()
    
    # assign parameters from file to layers
    
    def _load_parameters(self):
        
        self.old_state_dict = torch.load(f = self.param_path,
                                         map_location = torch.device('cpu'))
        
        # make a copy to use load_state_dict() method later
        
        state_dict = self.state_dict()
        
        for key,value in self.old_state_dict.items():
            
            # skip layer 19
            
            if key[12:14].isdigit() and int(key[12:14]) == 19:
                continue
            
            parameter_name = self._get_parameter_name(key)
            
            state_dict[parameter_name] = value
        
        self.load_state_dict(state_dict)
        
    def _get_parameter_name(self,key):
        
        # get layer number
        
        if key[12:14].isdigit():
            layer_num = int(key[12:14])
        else:
            layer_num = int(key[12])
        
        # get sub-layer number. 0 means conv layer and 1 means batch norm layer
        
        sub_layer_num = int(key.split('.')[2])
        
        # get parameter type
        
        parameter_type = key.split('.')[-1]
        
        """
        map layer number and sub-layer number to stage number and
        sub-stage number. Note that only convolutional layers and batch
        normalization layers have parameters
        """
        
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
        
        # flatten from N x 1024 x 1 x 1 to N x 1024
        
        x = torch.flatten(input = x,
                          start_dim = 1)
        
        return x

if __name__ == '__main__':
    
    import random
    
    param_path = 'mx-h64-1024_0d3-1.17.pkl'
    
    net = Net(param_path)
    
    idx = random.randint(0,15)
    
    test_feat_map1 = net.old_state_dict['module.layer1.0.weight'][idx]
    
    print('Example kernel from old state dict:\n{}\n'.format(test_feat_map1))
    
    test_feat_map2 = net.state_dict()['stage1.0.weight'][idx]
    
    print('Example kernel from new state dict:\n{}\n'.format(test_feat_map2))
    
    x = torch.randn((1,1,256,256))
    
    print('Example input:\n{}\n'.format(x))
    
    y = net(x)
    
    print('Output:\n{}\n'.format(y))
