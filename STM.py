import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class CMM_Channel_wise2D(nn.Module):
    def __init__(self,planes,stride):
        super(CMM_Channel_wise2D, self).__init__()
        self.Channel_wise_2d = nn.Conv3d(in_channels=planes,out_channels=planes,
                                         kernel_size=(1,3,3),groups=planes,
                                         padding=(0,1,1),stride=(1,stride,stride),
                                         bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)  # 定义一个逐点卷积进行通道融合
        self.bn1 = nn.BatchNorm3d(planes)

        #Input_channel:planes ==> output_channels:planes
        #============================================================================
        """定义CSTM模块"""
        # 定义2D的空间卷积
        #定义逐通道的1D时域卷积，按照文章中所述进行初始化，前1/4为[1,0,0],后1/4为[0,0,1],另一半为[0,1,0]
        self.channel_wise_1D = nn.Conv3d(in_channels=planes,out_channels=planes,
                                         kernel_size=(3,1,1),groups=planes,
                                         padding=(1,0,0),stride=(1,1,1),
                                         bias=False)
        #注意padding和stride的设置使得featuremap的尺寸受到变量stride的控制。
        # 定义2D空间卷积,用Imagenet进行初始化
        self.conv2 = nn.Conv3d(in_channels=planes,out_channels=planes,
                                kernel_size=(1,3,3),stride=(1,stride,stride),
                                padding=(0,1,1),bias=False)

        """定义CMM模块"""
        self.CMM_2d = nn.Conv3d(in_channels=planes,out_channels=planes/16,
                                kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),
                                bias=False)

        self.channel_2D_1 = self.CMM_Channel_wise2D(planes/16,stride)

        #============================================================================
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)  # 再一次进行3逐点卷积扩充通道
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)  # 在一次进行3DBN
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  #保留住残差

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # out ==> [batch_size,C,T,W,H]

        out = self.channel_wise_1D(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            #降采样的目的是是为下一步将residual与残差块卷积计算出来的特征图进行相加做准备
            #因为经过残差卷积后通道数量已经比residual多了，因此需要进行处理。
            residual = self.downsample(residual)

        out += residual  # 加上残差
        out = self.relu(out)  # 再进行激活一次

        return out

    def CMM_Channel_wise2D(self,planes,stride):

        return nn.Sequential(nn.Conv3d(in_channels=planes,out_channels=planes,
                                       kernel_size=(1,3,3),stride=(1,stride,stride),
                                       padding=(0,1,1),groups=planes))


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=174):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)# 定义Relu激活,输出featuremap的特征图的通道数为64
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        #此处self._make_layer函数的参数（64,128,256,512）
        # 表示的是bottleneck连接中第二个卷积层（3X3）卷积的输入和输出通道数
        #且bottleneck的第三个卷积（1X1的通道卷积的输出通道数量为，expansionX(64,128,256,512)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.new_fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #对所有的卷积层使用凯明初始化
            elif isinstance(m, nn.BatchNorm3d):#所有的BN层采用gamma=1,beta=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        # 只有每一组卷积的第一个bottneck连接需要用到downsample
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #此时featuremap为64
        print("maxpool.shape",x.size())

        x = self.layer1(x)
        print("layer1_out.size:",x.size())
        x = self.layer2(x)
        print("layer2_out.size:",x.size())
        x = self.layer3(x)
        print("layer3_out.size:",x.size())
        x = self.layer4(x)
        print("layer4_out.size:",x.size())
        x = x.transpose(1, 2).contiguous() #从此处开始
        x = x.view((-1,) + x.size()[2:])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.new_fc(x)
        print("out.size():",x.size())

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 based model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],**kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'],model_dir='./pretrained/')  # 加载Imagenet预训练的模型参数
    layer_name = list(checkpoint.keys())  # checkpoint是一个字典{'layer_name','权重tensor'}
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  # 扩充维度，很重要！
    model.load_state_dict(checkpoint, strict=False)
    return model

if __name__ == '__main__':
    model = resnet50()
    Input = torch.randn([1,3,8,224,224]) #N,C,T,W,H
    out = model(Input)
