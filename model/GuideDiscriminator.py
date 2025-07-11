import torch
from torch import nn

class GuideDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(GuideDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        # Construct three discriminator models
        self.models = nn.ModuleList()
        self.score_models = nn.ModuleList()
        for i in range(3):
            self.models.append(
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512)
                )
            )
            self.score_models.append(
                nn.Sequential(
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    # Compute the MSE between model output and scalar gt
    def compute_loss(self, x, gt):
        _, outputs = self.forward(x)

        loss = sum([torch.mean((out - gt) ** 2) for out in outputs])
        return loss

    def forward(self, x, x_low=None):
        outputs = []
        feats = []
        feats_low = []
        for i in range(len(self.models)):
            feats.append(self.models[i](x))
            outputs.append(self.score_models[i](self.models[i](x)))
            x = self.downsample(x)
        if x_low != None:
            for j in range(len(self.models) - 1):
                feats_low.append(self.models[j](x_low))  # `16*16, 8*8
                x_low = self.downsample(x_low)

        self.upsample = nn.Upsample(size=(feats[0].size()[2], feats[0].size()[3]), mode='nearest')
        feat = feats[0]
        if x_low != None:
            for i in range(1, len(feats)):
                feat = feat + self.upsample(feats_low[i - 1]) + self.upsample(feats[i])
        else:
            for i in range(1, len(feats)):
                feat = feat + self.upsample(feats[i])
        return feat, outputs