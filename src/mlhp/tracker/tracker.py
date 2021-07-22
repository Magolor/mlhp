from ..utils import *
import numpy as np

class Metric(Serializable):
    module_name = "metric"
    module_abbr = "met"

    def __init__(self, name=None, value=None, compare=None):
        self.name = name; self.value = value; self.compare = compare

# assert False

# class Tracker():
#     def __init__(self, title, DIR, registrations=[]):
#         self.title = title; self.DIR=DIR; Create(DIR); self.curve = {}; self.xlabel = {}; self.key = {}
#         self.data_file = os.path.join(self.DIR, "%s.dat"%self.title)
#         for key in registrations:
#             X, Y, b = key; self.curve[Y] = []; self.xlabel[Y] = X; self.key[Y] = b
#     def compare_func(self, b):
#         if b=='greater':
#             return lambda x: x
#         if b=='less':
#             return lambda x: -x
#         if b=='none':
#             return None
#         return b
#     def variable_profile(self, variable):
#         values = self.curve[variable]; key = self.compare_func(self.key[variable])
#         return values[np.argmax([key(x[1]) for x in values])][1] if len(values) else None
#     def profile(self):
#         profile = {}
#         for variable in self.curve.keys():
#             profile[variable] = self.variable_profile(variable)
#         return profile
#     def update(self, variable, value, time):
#         if value is None:
#             return False
#         assert(variable in self.curve)
#         self.curve[variable].append((time,value)); key = self.compare_func(self.key[variable])
#         return False if key is None else (key(value)>=max([key(x[1]) for x in self.curve[variable]]))
#     def load(self):
#         if os.path.exists(self.data_file):
#             self.__dict__.update(dict(torch.load(self.data_file)))
#     def plot(self):
#         for variable, values in self.curve.items():
#             if self.key[variable]!='none':
#                 values = sorted(values); X,Y = [v[0] for v in values], [v[1] for v in values]
#                 with Painter("%s: %s"%(self.title,variable), os.path.join(self.DIR,"%s.png"%variable)) as (fig,axe):
#                     plt.plot(X, Y); plt.xlabel(self.xlabel[variable]); plt.ylabel(variable)
#     def serialize(self):
#         SaveJSON(self.curve,os.path.join(self.DIR,"metrics.log"),indent=4)
#         SaveJSON(self.profile(),os.path.join(self.DIR,"profile.log"),indent=4)
#     def save(self):
#         torch.save(self.__dict__, self.data_file); self.plot(); self.serialize()
#     def __enter__(self):
#         return self
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.save()

class Tracker(Serializable):
    module_name = "tracker"
    module_abbr = "trc"
    def __init__(self, key_descriptors=list(), logger=Logger()):
        self.logger = logger; self.key_descriptors = key_descriptors
        self.curves = {k.name:np.array() for k in self.key_descriptors}