import torch
import torch.nn as nn
import torch.nn.functional as F

class ASP(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(ASP, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        w = self.attention(x) 
        mu = torch.sum(x * w, dim=2) 
        mu_expand = mu.unsqueeze(2)
        sg = torch.sqrt(torch.sum(w * ((x - mu_expand)**2), dim=2).clamp(min=1e-6))
        return torch.cat((mu, sg), 1)


class Resnet_block(nn.Module):
    expansion = 1
    
    def SimAM(self,X,lambda_p=1e-4):
        n = X.shape[2] * X.shape[3]-1
        d = (X-X.mean(dim=[2,3], keepdim=True)).pow(2)
        v = d.sum(dim=[2,3], keepdim=True)/n
        E_inv = d / (4*(v+lambda_p)) + 0.5
        return X * F.sigmoid(E_inv)
    
    def __init__(self, in_channels, out_channels, stride=(1,1), downsample=None, dropout_p=0.1):
        super(Resnet_block, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),
            stride=stride, padding=(1,1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3),
            stride=(1,1), padding=(1,1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample 
        self.dropout = nn.Dropout2d(p=dropout_p)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
       
        out = self.SimAM(out)
        
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out 


class Resnet34_model(nn.Module):
    def __init__(self, n_mels=80, embed_dim=512, dropout_p=0.2):
        super(Resnet34_model, self).__init__()
        self.in_channels = 64
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        self.input_adapt = nn.Unflatten(1, (1, n_mels))
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        
        self.layer1 = self._make_layer(Resnet_block, 64, num_blocks=3, stride=(1, 1))
        self.layer2 = self._make_layer(Resnet_block, 128, num_blocks=4, stride=(2, 1))
        self.layer3 = self._make_layer(Resnet_block, 256, num_blocks=6, stride=(2, 1))
        self.layer4 = self._make_layer(Resnet_block, 512, num_blocks=3, stride=(2, 1)) 

        self.freq_squeeze = nn.AdaptiveAvgPool2d((1, None)) 
        self.asp_pool = ASP(in_channels=512)
        self.embed_proj = nn.Linear(1024, embed_dim)
        self.bn_last = nn.BatchNorm1d(embed_dim)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_p=0.1):
        downsample = None
        if stride != (1, 1) or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_p=dropout_p))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_adapt(x)
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.freq_squeeze(x)
        x = x.squeeze(2) 
        x = self.asp_pool(x)
        embed = self.embed_proj(x)
        embed = self.bn_last(embed)
        return embed
    
class test_module1(nn.Module):
    def __init__(self):
        super(test_module1, self).__init__() 
        self.ResNet = Resnet34_model()
        
    def forward(self, x):
        out = self.ResNet(x)
        return out
        
