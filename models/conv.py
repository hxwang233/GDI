import torch
import torch.nn as nn
import torch.nn.init as init

class ConvNet(nn.Module):
    def __init__(self, height, width, in_channels, node_feats, n_class, channels, output_activation=torch.log_softmax):
        super(ConvNet, self).__init__()
        self.convs = nn.ModuleList()
        # 卷积输出大小=[（输入大小-卷积核（过滤器）大小+2*P）／步长]+1     H:10   W:3
        # 池化输出大小=[（输入大小-卷积核（过滤器）大小）／步长]+1
        for i in channels:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=i, stride=1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=0.2, inplace=False),
            )
            init.xavier_normal_(conv[0].weight)
            self.convs.append(conv)
            in_channels = i
        self.fc = nn.Linear(in_channels*height*width + node_feats, n_class)
        self.output_activation = output_activation
        init.xavier_normal_(self.fc.weight)
        return

    def forward(self, x, m):
        x = torch.unsqueeze(x, dim=1)
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, m], dim=1)
        x = self.output_activation(self.fc(x), dim=1)
        return x

