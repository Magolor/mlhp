from ..import_basic import *
from .file_helpers import *

class Serializable(object):
    module_name = "serializable"
    module_abbr = "ser"

    def __init__(self):
        pass

    def __str__(self):
        return str(self.__getstate__())

    def __getstate__(self):
        data = {}
        for key, value in self.__dict__.items():
            if hasattr(value,'module_name'):
                data[key] = value.__getstate__()
            else:
                data[key] = deepcopy(value)
        data.update({'module_name':self.module_name})
        return data

    def __setstate__(self, data):
        assert(data.pop('module_name')==self.module_name)
        for key, value in data.items():
            if hasattr(value,'module_name'):
                self.__dict__[key].__setstate__(value)
            else:
                self.__dict__[key] = deepcopy(value)
    
    def clone(self, **init_args):
        c =  self.__class__(**init_args);
        c.__setstate__(self.__getstate__())
        return c
    
    def save_json(self, path):
        SaveJSON(self.__getstate__(), path, indent=4)

    def load_json(self, path):
        self.__setstate__(LoadJSON(path))
        return self
    
    def save_pickle(self, path):
        SavePickle(self.__getstate__(), path)
    
    def load_pickle(self, path):
        self.__setstate__(LoadPickle(path))
        return self

    def __call__(self):
        return self

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class JSONSerializable(Serializable):
    module_name = "json_serializable"
    module_abbr = "jsn"

    def save(self, path):
        self.save_json(path)
    
    def load(self, path):
        return self.load_json(path)

class PickleSerializable(Serializable):
    module_name = "pickle_serializable"
    module_abbr = "pkl"

    def save(self, path):
        self.save_pickle(path)
    
    def load(self, path):
        return self.load_pickle(path)