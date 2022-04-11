import torch.nn as nn
import torch


class CNN(nn.Module):
    """CNN Resnet Model

    Arguments:
    __________
    device: torch.device
        device to run the model on
    """

    def __init__(self, num_classes=8):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        """Create a resnet layer

        Arguments:
        __________
        planes: int
            number of input channels
        blocks: int
            number of blocks in the layer
        stride: int
            stride of the layer

        Returns:
        ________
        layer: nn.Sequential
            layer of the model
        """
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the model

        Arguments:
        __________
        x: torch.Tensor
            input tensor

        Returns:
        ________
        out: torch.Tensor
            output tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    """Resnet Block

    Arguments:
    __________
    inplanes: int
        number of input channels
    planes: int
        number of output channels
    stride: int
        stride of the layer
    downsample: nn.Sequential
        downsample layer
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of the model

        Arguments:
        __________
        x: torch.Tensor
            input tensor

        Returns:
        ________
        out: torch.Tensor
            output tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
