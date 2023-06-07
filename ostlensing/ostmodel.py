import torch
import torch.nn as nn

from scattering_transform.scattering_transform import ScatteringTransform2d, Reducer
from scattering_transform.filters import FourierSubNetFilters, SubNet


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
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class OptimisableSTRegressor(nn.Module):
    def __init__(self,
                 size,
                 num_scales=6,
                 num_angles=6,
                 reduction=None,
                 hidden_sizes=(32, 32, 32),
                 output_size=1,
                 activation=nn.LeakyReLU,
                 seed=0
                 ):

        super(OptimisableSTRegressor, self).__init__()
        torch.manual_seed(seed)
        self.subnet = SubNet(hidden_sizes=(16, 16), activation=nn.LeakyReLU)
        self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet)
        self.st = ScatteringTransform2d(self.filters)
        self.reducer = Reducer(self.filters, reduction)
        self.batch_norm = nn.BatchNorm1d(self.reducer.num_outputs)
        self.regressor = MLP(input_size=self.reducer.num_outputs,
                             hidden_sizes=hidden_sizes,
                             output_size=output_size,
                             activation=activation)

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
