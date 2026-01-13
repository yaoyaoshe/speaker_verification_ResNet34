import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchaudio

# === 1. 将 ArcFace 类集成到此处 ===
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        return output

# === 2. ResNet 组件 (保持不变或微调) ===
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
        return X * torch.sigmoid(E_inv)
    
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
        # x input: (B, 80, T)
        x = self.input_adapt(x) # -> (B, 1, 80, T)
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
    def __init__(self, num_classes=None):
        super(test_module1, self).__init__() 
        self.ResNet = Resnet34_model()
        
        if num_classes is not None:
            self.arcface = ArcFace(in_features=512, out_features=num_classes)
        else:
            self.arcface = None
        
        # 初始化重采样器
        # Speed 1.1: 播放更快，音高更高 -> 将原始音频视为更高采样率，重采样回 16000
        # 16000 * 1.1 = 17600
        self.resample_fast = torchaudio.transforms.Resample(orig_freq=17600, new_freq=16000)
        
        # Speed 0.9: 播放更慢，音高更低 -> 将原始音频视为更低采样率，重采样回 16000
        # 16000 * 0.9 = 14400
        self.resample_slow = torchaudio.transforms.Resample(orig_freq=14400, new_freq=16000)

    def _compute_fbank(self, waveform, speed_ids=None):
        """
        内部计算 Fbank，并处理变长问题
        waveform: (B, T)
        Returns: (B, 80, T_frames)
        """
        fbanks = []
        for i in range(waveform.size(0)):
            wav = waveform[i].unsqueeze(0) # (1, T)
            
            # GPU并行重采样
            if speed_ids is not None:
                s_type = speed_ids[i].item()
                if s_type == 1: # 1.1x (加速 -> 变短)
                    wav = self.resample_fast(wav)
                elif s_type == 2: # 0.9x (减速 -> 变长)
                    wav = self.resample_slow(wav)
                # s_type == 0 不做处理
            
            # 提取特征
            f = torchaudio.compliance.kaldi.fbank(
                wav,
                sample_frequency=16000,
                num_mel_bins=80,
                frame_length=25.0, 
                frame_shift=10.0, 
                dither=0.0, 
                snip_edges=True
            )
            # f: (T_frames, 80)
            
            # CMVN (对每一条单独做)
            f = f - f.mean(dim=0, keepdim=True)
            
            # Transpose to (80, T_frames)
            f = f.transpose(0, 1) 
            fbanks.append(f)
            
        # === 新增：Padding 逻辑 ===
        # 1. 找到当前 Batch 中最大的帧数
        max_len = max([x.size(1) for x in fbanks])
        
        padded_fbanks = []
        for f in fbanks:
            # 计算需要 Pad 多少帧
            pad_len = max_len - f.size(1)
            if pad_len > 0:
                # F.pad 参数格式 (left, right, top, bottom)
                # 我们只 Pad 最后一个维度 (Time) 的右侧
                f = F.pad(f, (0, pad_len), mode='constant', value=0.0)
            padded_fbanks.append(f)
            
        return torch.stack(padded_fbanks).to(waveform.device)
    
    def forward(self, x, label=None, speed_ids=None):
        # 自动判断输入类型
        if x.dim() == 2:
            # 训练时传入 Raw Waveform + speed_ids
            x = self._compute_fbank(x, speed_ids) # -> (B, 80, T)
        elif x.dim() == 4:
            x = x.squeeze(1) 
        
        embedding = self.ResNet(x)
        
        if label is not None and self.arcface is not None:
            output = self.arcface(embedding, label)
            return output
        
        return embedding