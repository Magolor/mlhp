from timm import create_model

class Timm(nn.Module):
    module_name = "timm"
    module_abbr = "tim"

    def __init__(self, module_name='timm', module_abbr='tim', model_type="", **model_args):
        super(Timm, self).__init__()
        self.module_name = module_name
        self.module_abbr = module_abbr
        self.net = create_model(model_type, **model_args)

    def forward(self, data):
        return self.net(data)