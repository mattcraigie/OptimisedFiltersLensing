import torch
import torch.nn as nn

from transformers import ResNetModel, ResNetConfig

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import FourierSubNetFilters, SubNet
from .training import batch_apply


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.LeakyReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class OptimisableSTRegressor(nn.Module):
    def __init__(self,
                 size,
                 num_scales=4,
                 num_angles=4,
                 reduction=None,
                 hidden_sizes=(32, 32, 32),
                 output_size=1,
                 activation=nn.GELU
                 ):
        super(OptimisableSTRegressor, self).__init__()
        self.subnet = SubNet(hidden_sizes=(32, 32, 32), activation=activation)
        self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet)
        self.st = ScatteringTransform2d(self.filters, clip_sizes=[size // 2 ** i for i in range(num_scales)])
        self.reducer = Reducer(self.filters, reduction)
        self.batch_norm = nn.BatchNorm1d(self.reducer.num_outputs)
        self.regressor = MLP(input_size=self.reducer.num_outputs,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activation=nn.LeakyReLU)
        self.device = None

    def forward(self, x):
        self.filters.update_filters()
        self.st.clip_filters()
        x = self.reducer(self.st(x))
        x = x.mean(1)  # 'channel' mean, i.e. mean across all patches for the same cosmology
        x = self.batch_norm(x)
        return self.regressor(x)

    def to(self, device):
        super(OptimisableSTRegressor, self).to(device)
        self.st.to(device)
        self.device = device
        return self


class PreCalcRegressor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes=(32, 32, 32),
                 output_size=1,
                 activation=nn.LeakyReLU
                 ):
        super(PreCalcRegressor, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.regressor = MLP(input_size=input_size,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activation=activation)

    def forward(self, x):
        x = x.mean(1)  # todo: I should just put this in the precalc script!
        x = self.batch_norm(x)
        return self.regressor(x)


class ResNetRegressor(PreCalcRegressor):
    def __init__(
            self,
            pretrained=True,
            regressor_hiddens=(2048, 512, 512),
            regressor_outputs=1,
            regressor_activations=nn.LeakyReLU,
    ):
        super(ResNetRegressor, self).__init__(
            2048,
            hidden_sizes=regressor_hiddens,
            output_size=regressor_outputs,
            activation=regressor_activations
        )

        self.channel_upsample = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        if pretrained:
            self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50')
        else:
            config = ResNetConfig()
            self.resnet = ResNetModel(config)

    def forward(self, x):
        batch_dim = x.shape[0]
        patch_dim = x.shape[1]
        size = x.shape[2]
        x = x.reshape(batch_dim * patch_dim, 1, size, size)
        x = self.channel_upsample(x)
        x = self.resnet(x).pooler_output
        x = x.reshape(batch_dim, patch_dim, -1)
        return super(ResNetRegressor, self).forward(x)
