from ..utils import *

class Handle(PickleSerializable):
    module_name = "handle"
    module_abbr = "hdl"

    def __init__(self, handle=None, priority=None):
        self.handle = handle; self.priority = priority

    def __getitem__(self, i):
        return [self.handle, self.priority][i]

class Logger(PickleSerializable):
    module_name = "logger"
    module_abbr = "log"
    ALL = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5
    MAX = 65535
    __protocol__ = {
        'out': sys.stdout,
        'err': sys.stderr,
        'in':  sys.stdin,
    }

    def __init__(self, handles=list()):
        self.h = {}
        for handle in handles:
            self.add_handle(handle)

    def add_handle(self, handle):
        handle = tuple(handle)
        (t,h), l = handle[0].split(':'), handle[1]
        htype = ['sys','w','a']
        assert (t in htype), f"Supported Handle Types: {htype}"
        if t in 'sys':
            sysh = ['out','err','in','ret','default']
            assert (h in sysh), f"Supported Handle Types of 'sys': {sysh}"
            self.h[handle[0]] = {
                'handle': h,
                'fileargs': h,
                'level': l,
            }
        else:
            CreateFile(h)
            self.h[handle[0]] = {
                'handle': open(h,t),
                'fileargs': (h,t),
                'level': l,
            }
    
    def log(self, *args, **kwargs):
        level = kwargs.pop('level') if 'level' in kwargs else Logger.MAX
        L = []
        for key in self.h.keys():
            h, l = self.h[key]['handle'], self.h[key]['level']
            if level >= l:
                if h in Logger.__protocol__:
                    print(*args, file=Logger.__protocol__[h], flush=True, **kwargs)
                elif h=='default':
                    print(*args, flush=True, **kwargs)
                elif h=='ret':
                    raise NotImplementedError
                    # L.append()
                else:
                    print(*args, file=h, flush=True, **kwargs)
    
    def log_handles(self, *args, **kwargs):
        handles = kwargs.pop('handles')
        for key in handles:
            assert (key in self.get_handles()), f"Some keys do not exist in logger handles!"
            h = self.h[key]
            if h in Logger.__protocol__:
                print(*args, file=Logger.__protocol__[self.h], flush=True, **kwargs)
            elif h=='default':
                print(*args, flush=True, **kwargs)
            else:
                print(*args, file=h, flush=True, **kwargs)
    
    def get_handles(self):
        return list(self.h.keys())

    def get_level(self, handle):
        return self.h[handle]['level']
    
    def set_level(self, handle, level):
        self.h[handle]['level'] = level
    
    def __getstate__(self):
        tmp = [None for _ in range(len(self.h))]
        for i,key in enumerate(self.h.keys()):
            tmp[i], self.h[key]['handle'] = self.h[key]['handle'], tmp[i]
        s = super(Logger, self).__getstate__()
        for i,key in enumerate(self.h.keys()):
            tmp[i], self.h[key]['handle'] = self.h[key]['handle'], tmp[i]
        return s

    def __setstate__(self, data):
        super(Logger, self).__setstate__(data)
        for key in self.get_handles():
            self.h[key]['handle'] = open(self.h[key]['fileargs'][0],'a') if isinstance(self.h[key]['fileargs'],tuple) else self.h[key]['fileargs']

NONE_LOGGER = Logger()
CONSOLE_LOGGER = Logger(handles=[Handle("sys:out",Logger.ALL)])
ERROR_LOGGER = Logger(handles=[Handle("sys:err",Logger.ALL)])