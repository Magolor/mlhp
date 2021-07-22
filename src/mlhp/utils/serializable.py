from ..import_basic import *
from .file_helpers import *

class Serializable(object):
    module_name = "serializable"
    module_abbr = "ser"

    def __init__(self):
        pass

    def __str__(self):
        return str(self.__get_state__())

    def __get_state__(self):
        data = {}
        for key, value in self.__dict__.items():
            if hasattr(value,'module_name'):
                data[key] = value.__get_state__()
            else:
                data[key] = deepcopy(value)
        data.update({'module_name':self.module_name})
        return data

    def __set_state__(self, data):
        assert(data.pop('module_name')==self.module_name)
        for key, value in data.items():
            if hasattr(value,'module_name'):
                self.__dict__[key].__set_state__(value)
            else:
                self.__dict__[key] = deepcopy(value)
    
    def save_json(self, path):
        SaveJSON(self.__get_state__(), path)

    def load_json(self, path):
        self.__set_state__(LoadJSON(path))
        return self
    
    def save_pickle(self, path):
        SavePickle(self.__get_state__(), path)
    
    def load_pickle(self, path):
        self.__set_state__(LoadPickle(path))
        return self