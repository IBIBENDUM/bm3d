import torch
import torch.nn as nn
import torch.nn.functional as F

def calculatePsnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
    
    def forward(self, pred, target):
        return -calculatePsnr(pred, target, self.max_val)

class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels, outChannels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outChannels, outChannels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )


    def forward(self, inputTensor):
        return self.conv(inputTensor)


class EncoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(inChannels, outChannels)
        )


    def forward(self, inputTensor):
        return self.encoder(inputTensor)


class DecoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.upSample = nn.ConvTranspose2d(
            in_channels=inChannels,
            out_channels=inChannels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(inChannels, outChannels)


    def forward(self, inputTensor, skipConnection):
        upSampled = self.upSample(inputTensor)
        heightDiff = skipConnection.size()[2] - upSampled.size()[2]
        widthDiff = skipConnection.size()[3] - upSampled.size()[3]

        upSampled = F.pad(
            upSampled,
            [
                widthDiff // 2,
                widthDiff - widthDiff // 2,
                heightDiff // 2,
                heightDiff - heightDiff // 2,
            ],
        )

        merged = torch.cat([skipConnection, upSampled], dim=1)

        return self.conv(merged)


class UNet(nn.Module):
    def __init__(
        self, inChannels=1, outChannels=1, featureSizes=[64, 128, 256, 512, 1024]
    ):
        super().__init__()
        self.inputConv = DoubleConv(inChannels, featureSizes[0])

        self.downBlocks = nn.ModuleList(
            [
                EncoderBlock(featureSizes[i], featureSizes[i + 1])
                for i in range(len(featureSizes) - 1)
            ]
        )

        self.upBlocks = nn.ModuleList(
            [
                DecoderBlock(featureSizes[i], featureSizes[i - 1])
                for i in range(len(featureSizes) - 1, 0, -1)
            ]
        )

        self.finalConv = nn.Conv2d(featureSizes[0], outChannels, kernel_size=1)


    def forward(self, inputTensor):
        skipConnections = []
        tensor = self.inputConv(inputTensor)

        for down in self.downBlocks:
            skipConnections.append(tensor)
            tensor = down(tensor)

        for up, skip in zip(self.upBlocks, reversed(skipConnections)):
            tensor = up(tensor, skip)

        return self.finalConv(tensor)
