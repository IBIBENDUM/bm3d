import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.MaxPool2d(2),
            DoubleConv(inChannels, outChannels)
        )

    def forward(self, inputTensor):
        return self.encoder(inputTensor)


class DecoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels, dynamicPadding=True, paddingMode="constant"):
        super().__init__()
        self.upSample = nn.ConvTranspose2d(
            in_channels=inChannels,
            out_channels=inChannels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(inChannels, outChannels)
        self.dynamicPadding = dynamicPadding
        self.paddingMode = paddingMode


    def forward(self, inputTensor, skipConnection):
        upSampled = self.upSample(inputTensor)
        heightDiff = skipConnection.size()[2] - upSampled.size()[2]
        widthDiff = skipConnection.size()[3] - upSampled.size()[3]

        if self.dynamicPadding:
            upSampled = F.pad(
                upSampled,
               [
                    widthDiff // 2,
                    widthDiff - widthDiff // 2,
                    heightDiff // 2,
                    heightDiff - heightDiff // 2,
                ],
                mode=self.paddingMode
            )

        merged = torch.cat([skipConnection, upSampled], dim=1)

        return self.conv(merged)


class UNet(nn.Module):
    def __init__(
        self,
        inChannels=1,
        outChannels=1,
        featureSizes=[64, 128, 256, 512, 1024],
        dynamicPadding=True,
        paddingMode="constant"
    ):
        super().__init__()
        self.inputConv = DoubleConv(inChannels, featureSizes[0])

        self.dynamicPadding = dynamicPadding
        self.downsampleFactor = 2 ** (len(featureSizes) - 1)

        self.paddingMode = paddingMode

        self.downBlocks = nn.ModuleList(
            [
                EncoderBlock(featureSizes[i], featureSizes[i + 1])
                for i in range(len(featureSizes) - 1)
            ]
        )

        self.upBlocks = nn.ModuleList(
            [
                DecoderBlock(featureSizes[i], featureSizes[i - 1], dynamicPadding, paddingMode)
                for i in range(len(featureSizes) - 1, 0, -1)
            ]
        )

        self.finalConv = nn.Conv2d(featureSizes[0], outChannels, kernel_size=1)

    def padImage(self, tensor):
        _, _, h, w = tensor.size()
        factor = self.downsampleFactor
        hPad = (factor - h % factor) % factor
        wPad = (factor - w % factor) % factor

        self.inputPad = (
            wPad // 2,
            wPad - wPad // 2,
            hPad // 2,
            hPad - hPad // 2,
        )

        return F.pad(tensor, self.inputPad, mode=self.paddingMode)

    def unpadImage(self, tensor):
        _, _, h, w = tensor.size()
        wPadLeft, wPadRight, hPadtop, hPadBottom = self.inputPad
        unpaddedTensor = tensor[:, :, hPadtop : h - hPadBottom, wPadLeft : w - wPadRight] 

        return unpaddedTensor

    def forward(self, inputTensor):
        if self.dynamicPadding is False:
            inputTensor = self.padImage(inputTensor)

        skipConnections = []
        tensor = self.inputConv(inputTensor)

        for down in self.downBlocks:
            skipConnections.append(tensor)
            tensor = down(tensor)

        for up, skip in zip(self.upBlocks, reversed(skipConnections)):
            tensor = up(tensor, skip)

        output = self.finalConv(tensor) 

        if self.dynamicPadding is False:
            output = self.unpadImage(output)

        return output
