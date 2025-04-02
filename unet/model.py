import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

class DoubleConv(nn.Module):
    """
    Double convolution block
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
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
    """
    Encoder block
    Reduces the spatial dimensions of the input while increasing the number of feature channels
    Max pooling -> DoubleConv
    """
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.encoder = nn.Sequential(
            # Reduce spatial dimensions by half
            nn.MaxPool2d(2),
            DoubleConv(inChannels, outChannels),
        )

    def forward(self, inputTensor):
        return self.encoder(inputTensor)


class DecoderBlock(nn.Module):
    """
    Decoder Block with upsampling and concatenation with skip connections
    """
    def __init__(self, inChannels, outChannels, dynamicPadding=True, paddingMode="constant"):
        super().__init__()
        # Transpose convolution for upsampling
        self.upSample = nn.ConvTranspose2d(
            in_channels=inChannels,
            out_channels=inChannels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(inChannels, outChannels)
        self.dynamicPadding = dynamicPadding
        self.paddingMode = paddingMode

    def applyDynamicPadding(self, upSampled, skipConnection):
        # Calculate difference in height and width between the skip connection and upsampled tensor
        heightDiff = skipConnection.size(2) - upSampled.size(2)
        widthDiff = skipConnection.size(3) - upSampled.size(3)

        upSampled = F.pad(
            upSampled,
            [
                widthDiff // 2,
                widthDiff - widthDiff // 2,
                heightDiff // 2,
                heightDiff - heightDiff // 2,
            ],
            mode=self.paddingMode,
        )

        return upSampled

    def forward(self, inputTensor, skipConnection):
        # Upsample the input tensor
        upSampled = self.upSample(inputTensor)

        # If dynamic padding is enabled, adjust padding during the forward pass
        if self.dynamicPadding:
            upSampled = self.applyDynamicPadding(upSampled, skipConnection)

        # Concatenate the skip connection with the upsampled tensor
        merged = torch.cat([skipConnection, upSampled], dim=1)

        return self.conv(merged)


class UNet(nn.Module):
    """
    Default U-Net architecture
    The model consists of an encoder and a decoder part,
    with skip connections between corresponding encoder and decoder layers
    """
    def __init__(
        self,
        inChannels=1,
        outChannels=1,
        featureSizes=[64, 128, 256, 512, 1024],
        dynamicPadding=True,
        paddingMode="constant"
    ):
        super().__init__()

        # Initial convolution layer
        self.inputConv = DoubleConv(inChannels, featureSizes[0])

        # Padding and downsampling factors
        self.dynamicPadding = dynamicPadding
        self.downsampleFactor = 2 ** (len(featureSizes) - 1)
        self.paddingMode = paddingMode

        # Create downsampling blocks
        self.downBlocks = nn.ModuleList(
            [
                EncoderBlock(featureSizes[i], featureSizes[i + 1])
                for i in range(len(featureSizes) - 1)
            ]
        )

        # Create upsampling blocks
        self.upBlocks = nn.ModuleList(
            [
                DecoderBlock(featureSizes[i], featureSizes[i - 1], dynamicPadding, paddingMode)
                for i in range(len(featureSizes) - 1, 0, -1)
            ]
        )

        # Final convolution layer for output
        self.finalConv = nn.Conv2d(featureSizes[0], outChannels, kernel_size=1)

    def padImage(self, tensor):
        """
        Pad input image to ensure its dimensions are divisible by the downsampling factor
        """
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
        """
        Crop tensor after padding to restore the original size
        """
        hTarget = tensor.size(2) - self.inputPad[2] - self.inputPad[3]
        wTarget = tensor.size(3) - self.inputPad[0] - self.inputPad[1]

        return CenterCrop((hTarget, wTarget))(tensor)

    def forward(self, inputTensor):
        # Apply static padding if dynamic padding is not used
        if not self.dynamicPadding:
            inputTensor = self.padImage(inputTensor)

        skipConnections = []
        tensor = self.inputConv(inputTensor)

        # Downward pass through the encoder blocks
        for down in self.downBlocks:
            skipConnections.append(tensor)
            tensor = down(tensor)

        # Upward pass through the decoder blocks, with skip connections
        for up, skip in zip(self.upBlocks, reversed(skipConnections)):
            tensor = up(tensor, skip)

        # Final convolution to get the output
        output = self.finalConv(tensor) 

        # Remove padding if static padding was used
        if self.dynamicPadding is False:
            output = self.unpadImage(output)

        return output
