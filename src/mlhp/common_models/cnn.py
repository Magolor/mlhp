from ..utils import *
from ..import_torch import *

class CNN(nn.Module):
    module_name = "mlp"
    module_abbr = "mlp"
    
    def __init__(
        self,
        layer_sizes = [1,1],
        bias = True,
        activations = [nn.Identity()],
    ):
        super(CNN, self).__init__()
        assert (len(activations)==len(layer_sizes)-1)
        self.layers = nn.Sequential(); self.num_features = layer_sizes[0]; self.num_classes = layer_sizes[-1]
        for i,layer_size in enumerate(layer_sizes[1:], 1):
            self.layers.add_module('fc%02d'%i,nn.Linear(layer_sizes[i-1],layer_size,bias=bias))
            self.layers.add_module('activation%02d'%i,activations[i-1])
    
    def forward(self, data):
        return self.layers(data.reshape(data.shape[0],self.num_features))