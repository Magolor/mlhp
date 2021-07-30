from ..utils import *
from ..import_torch import *
from timm import create_model
from ..torch_serializable.torch_serializable import TorchSerializable
from torchvision.models._utils import IntermediateLayerGetter

class TimmBackbone(nn.Module, TorchSerializable):
    module_name = "timm"
    module_abbr = "tim"

    def __init__(self, model_type="", module_name='timm', module_abbr='tim',
        input_layer_getter=lambda n:n.conv1, modify_input_channels=None,
        output_layer_name="fc", embedding_dim=2048,
        **model_args
    ):
        super(TimmBackbone, self).__init__()
        self.module_name = module_name
        self.module_abbr = module_abbr
        self.net = create_model(model_type, **model_args)

        layer = input_layer_getter(self.net); device = layer.weight.device
        if modify_input_channels and modify_input_channels!=layer.in_channels:
            layer.in_channels = modify_input_channels
            shape = layer.weight.shape; shape = torch.Size([shape[0],modify_input_channels,shape[2],shape[3]])
            layer.weight = nn.Parameter(torch.zeros(shape)).to(device)
            nninit.xavier_uniform_(layer.weight)
        layer = input_layer_getter(self.net)
        self.in_channels = layer.in_channels
        
        self.net = IntermediateLayerGetter(self.net, return_layers={output_layer_name:"out"})
        self.net._modules[output_layer_name] = nn.Linear(self.net._modules[output_layer_name].in_features,embedding_dim,bias=True)
        self.net.to(device)

    def forward(self, data):
        return self.net(data)['out']