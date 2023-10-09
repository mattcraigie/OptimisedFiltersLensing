import torch
import torch.nn as nn

from transformers import ResNetModel, ResNetConfig

from scattering_transform.scattering_transform import ScatteringTransform2d
from scattering_transform.filters import FourierSubNetFilters, SubNet, FourierDirectFilters, TrainableMorlet
from scattering_transform.reducer import Reducer


# ~~~ General Models ~~~ #

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        if hidden_sizes is None:
            self.model = nn.Sequential(
                nn.Linear(input_size, output_size)
            )
        else:
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


# ~~~ Model Wrappers ~~~ #

# The forward methods input a flattened batch/patch tensor with an input shape of (batch * patch, 1, size, size).
# The forward methods output a vector of summary statistics of shape (batch * patch, num_outputs)

class ResNetWrapper(nn.Module):
    def __init__(self, pretrained_model=None):
        super(ResNetWrapper, self).__init__()

        # pretrained model is in form from huggingface e.g. 'microsoft/resnet-18'
        self.channel_upsample = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        if pretrained_model is None:
            config = ResNetConfig()  # these are resnet-50 defaults
            self.resnet = ResNetModel(config)
        else:
            self.resnet = ResNetModel.from_pretrained(pretrained_model)

        # get number of outputs by running a dummy pass
        with torch.no_grad():
            # pooler output shape is (batch, outputs, 1, 1)
            self.num_outputs = self.resnet(torch.zeros(1, 3, 128, 128)).pooler_output.shape[1]

    def forward(self, x):
        batch, patch, size, _ = x.shape
        x = x.reshape(batch * patch, 1, size, size)
        x = self.channel_upsample(x)
        x = self.resnet(x).pooler_output
        return x.reshape(batch, patch, self.num_outputs)



class OSTWrapper(nn.Module):
    def __init__(self,
                 size=128,
                 num_scales=4,
                 num_angles=4,
                 reduction=None,
                 ost_type=None,
                 subnet_hiddens=(128, 128),
                 enforce_symmetry=True,
                 subnet_activations=nn.SiLU,
                 scale_invariant=False,
                 init_morlet=False,
                 freeze_filters=False,
                 scaled_sizes=None,
                 periodic=False,
                 ):
        super(OSTWrapper, self).__init__()

        assert ost_type in ['subnet', 'direct', 'trainable_morlet'], 'Invalid OST type'

        if ost_type == 'subnet':
            subnet_inputs = 2 if scale_invariant else 3
            self.subnet = SubNet(num_ins=subnet_inputs, hidden_sizes=subnet_hiddens, num_outs=1,
                                 activation=subnet_activations)
            self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet,
                                                scale_invariant=scale_invariant, init_morlet=init_morlet,
                                                clip_sizes=scaled_sizes, periodic=periodic)
            if freeze_filters:
                for param in self.subnet.parameters():
                    param.requires_grad = False

        elif ost_type == 'direct':
            self.filters = FourierDirectFilters(size, num_scales, num_angles, init_morlet=init_morlet)

        elif ost_type == 'trainable_morlet':
            self.filters = TrainableMorlet(size, num_scales, num_angles, scale_invariant=scale_invariant,
                                           enforce_symmetry=enforce_symmetry)

        self.st = ScatteringTransform2d(self.filters)
        self.reducer = Reducer(self.filters, reduction, normalise_s2=True)
        self.num_outputs = self.reducer.num_outputs

    def forward(self, x):
        self.filters.update_filters()
        return self.reducer(self.st(x))

    def to(self, device):
        super(OSTWrapper, self).to(device)
        self.filters.to(device)
        self.st.to(device)
        self.reducer.to(device)
        self.device = device
        return self





# ~~~ Model Dict ~~~ #
model_dict = {'resnet': ResNetWrapper, 'ost': OSTWrapper}


# ~~~ Regression Model ~~~ #

class ModelRegressor(nn.Module):
    # This is for models that output features
    def __init__(self,
                 regressor_type='precalc',
                 model_type=None,
                 model_kwargs=None,
                 regressor_inputs=None,
                 regressor_hiddens=None,
                 regressor_outputs=1,
                 regressor_activations=nn.ReLU,
                 regressor_batchnorm=False,
                 random_seed=0,
                 ):

        super(ModelRegressor, self).__init__()

        torch.manual_seed(random_seed)

        if regressor_type == 'patch':
            self.model = model_dict[model_type](**model_kwargs)
            regressor_inputs = self.model.num_outputs
            self.patch = True
        else:
            assert regressor_inputs is not None, 'Must specify regressor_inputs for precalc'
            self.model = nn.Identity()
            self.patch = False

        if regressor_batchnorm:
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(regressor_inputs),
                MLP(regressor_inputs, regressor_hiddens, regressor_outputs, regressor_activations)
            )
        else:
            self.regressor = MLP(regressor_inputs, regressor_hiddens, regressor_outputs, regressor_activations)

    def to(self, device):
        super(ModelRegressor, self).to(device)
        self.model.to(device)
        self.device = device
        return self

    def forward(self, x):
        x = self.model(x)
        if self.patch:
            x = x.mean(1)  # average over all patches that have the same cosmology
        x = self.regressor(x)
        return x  # final output should have shape (batch, regressor_outputs)

    def save(self, path):
        torch.save(self.state_dict(), path)

