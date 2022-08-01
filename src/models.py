import torch.nn as nn

# Creating a CNN class
class Custom_0(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self,num_classes=10):
        super(Custom_0, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)


        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,padding=3)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        # self.fc1 = nn.Linear(1600, 128)
        self.fc1 = nn.Linear(61504, 128)
        self.fc2 = nn.Linear(128, num_classes)
            
        ### Cross entroy loss already includes log soft max layer

    # Progresses data across layers    
    def forward(self, x):
        ### CONV 1
        out = self.conv_layer1(x)
        out = self.relu(out)
        out = self.dropout(out)

        ### CONV 2
        out = self.conv_layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        ### MAX POOL
        out = self.max_pool(out)
        
        ### CONV 3
        out = self.conv_layer3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        ### CONV 4
        out = self.conv_layer4(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        ### MAX POOL
        out = self.max_pool(out)

        ### FULLY CONNECTED LAYERS                
        out = out.reshape(out.size(0), -1)
        # out = out.view(out.size(0), -1)
        # print(out.shape)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out