from ..utils import *
import torch

def SaveTorch(obj, path, **args):
    CreateFile(path); torch.save(obj, PathToString(path), **args)

def LoadTorch(path, **args):
    assert (ExistFile(path)), "path '{0}' does not exist".format(path)
    return torch.load(PathToString(path), **args)

class TorchSerializable(Serializable):
    module_name = "torch_serializable"
    module_abbr = "pth"
    
    def save_torch(self, path):
        SaveTorch(self.__getstate__(), path)
    
    def load_torch(self, path):
        self.__setstate__(LoadTorch(path))
        return self

    def save(self, path):
        self.save_torch(path)
    
    def load(self, path):
        return self.load_torch(path)