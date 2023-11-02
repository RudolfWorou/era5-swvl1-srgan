import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        self.prelu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)


class Generator(torch.nn.Module):
    def __init__(
        self,
        input_channels=1,
        upscale_factor=4,
        num_channels=64,
        num_residual_blocks=16,
    ):
        super(Generator, self).__init__()

        self.upscale_factor = upscale_factor

        # Initial convolution block
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                num_channels,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            torch.nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(num_channels))
        self.res_blocks = torch.nn.Sequential(*res_blocks)

        # Upsampling blocks
        upsampling_blocks = []
        for _ in range(int(upscale_factor / 2)):
            upsampling_blocks.append(UpsampleBlock(num_channels, num_channels))
        self.upsampling_blocks = torch.nn.Sequential(*upsampling_blocks)

        # Output layer
        self.conv2 = torch.nn.Conv2d(
            num_channels, input_channels, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = torch.add(x, residual)
        x = self.upsampling_blocks(x)
        x = self.conv2(x)
        return (torch.tanh(x) + 1) / 2


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=1, low_res_size=32):
        super().__init__()

        self.blocks = []

        L = [input_channels, 64, 64, 128, 128, 256, 256, 512, 512]
        for i in range(8):
            self.blocks.append(
                torch.nn.Sequential(
                    *[
                        torch.nn.Conv2d(
                            L[i],
                            L[i + 1],
                            kernel_size=3,
                            stride=(i % 2) + 1,
                            padding=1,
                        ),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            )

        self.blocks = torch.nn.ModuleList(self.blocks)
        self.last_block = torch.nn.Sequential(
            *[
                torch.nn.Linear(low_res_size // 4, 1024),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(1024, 1),
                torch.nn.Sigmoid(),
            ]
        )

    def forward(self, x):
        for i in range(8):
            x = self.blocks[i](x)
        x = self.last_block(x)
        return x
