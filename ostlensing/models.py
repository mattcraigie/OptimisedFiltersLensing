import torch
import torch.nn as nn

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import FourierSubNetFilters, SubNet
from .training import batch_apply

from torchvision.models import resnet18, vit_b_16



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
    def __init__(self,
                 model_output_size=512,
                 hidden_sizes=(512, 256, 128),
                 regression_output_size=1,
                 activation=nn.LeakyReLU,
                 ):
        super(ResNetRegressor, self).__init__(model_output_size,
                                              hidden_sizes=hidden_sizes,
                                              output_size=regression_output_size,
                                              activation=activation)

        resnet = resnet18()

        # change the first layer to input a single channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # change the last layer to output the correct size
        resnet.fc = nn.Linear(resnet.fc.in_features, model_output_size)
        self.resnet = resnet

    def forward(self, x):
        batch_dim = x.shape[0]
        patch_dim = x.shape[1]
        size = x.shape[2]
        x = x.reshape(batch_dim * patch_dim, 1, size, size)
        x = self.resnet(x)
        x = x.reshape(batch_dim, patch_dim, -1)
        return super(ResNetRegressor, self).forward(x)


class ViTRegressor(PreCalcRegressor):
    def __init__(self,
                 model_output_size=512,
                 hidden_sizes=(32, 32, 32),
                 regression_output_size=1,
                 activation=nn.LeakyReLU,
                 ):
        super(ViTRegressor, self).__init__(model_output_size,
                                           hidden_sizes=hidden_sizes,
                                           output_size=regression_output_size,
                                           activation=activation)

        vit = vit_b_16()

        # change the first layer to input a single channel


        # change the last layer to output the correct size
        vit.head = nn.Linear(vit.head.in_features, model_output_size)
        self.vit = vit

    def forward(self, x):
        batch_dim = x.shape[0]
        patch_dim = x.shape[1]
        size = x.shape[2]
        x = x.reshape(batch_dim * patch_dim, 1, size, size)
        x = self.vit(x)
        x = x.reshape(batch_dim, patch_dim, -1)
        return super(ViTRegressor, self).forward(x)
