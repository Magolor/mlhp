from ..utils import *
from ..plotter import *
import numpy as np

class Metric(PickleSerializable):
    module_name = "metric"
    module_abbr = "met"

    def __init__(self, name=None, key=lambda x:x):
        self.name = name; self.key = key.__code__
    
    def __getitem__(self, i):
        return [self.name, self.key][i]

    def __getstate__(self):
        tmp = self.key
        self.key = marshal.dumps(self.key)
        s = super(Metric, self).__getstate__()
        tmp, self.key = self.key, tmp
        return s
    
    def __setstate__(self, data):
        super(Metric, self).__setstate__(data)

class Tracker(PickleSerializable):
    module_name = "tracker"
    module_abbr = "trk"

    def __init__(self, title=None, path=None, metrics=list(), time_unit="epoch", plotter=Plotter()):
        self.title = title; self.path = path; self.time_unit = time_unit
        self.metrics = {metric.name:metric for metric in metrics}
        self.curves = {metric.name:[] for metric in metrics}
        self.plotter = plotter

    def metric_data(self, metric):
        return self.curves[metric]

    def metric_best(self, metric):
        key = types.FunctionType(self.metrics[metric].key, globals())
        curve = self.metric_data(metric=metric)
        if len(curve) > 0:
            best_ind = np.argmax([key(v[1]) for v in curve])
            return best_ind, curve[best_ind][0], curve[best_ind][1]
        else:
            return None
    
    def profile(self):
        prof = {}
        for metric in self.metrics:
            prof[metric]['num'] = len(self.metric_data(metric))
            prof[metric]['best'] = self.metric_best(metric)
        return prof
    
    def update(self, metric, time, value):
        key = types.FunctionType(self.metrics[metric].key, globals())
        self.curves[metric].append((time,value))
        return key(value) <= key(self.metric_best(metric)[2])

    def plot(self):
        for metric, curve in self.curves.items():
            curve = sorted(curve,key=lambda x:x[0]); X,Y = [v[0] for v in curve], [v[1] for v in curve]
            with self.plotter(title=metric,path=Folder(self.path)+metric+".png") as (fig,axe):
                plt.plot(X, Y); plt.xlabel(self.time_unit); plt.ylabel(metric)
    
    def reset(self, metric):
        self.curves[metric] = []
    
    def reset_all(self):
        self.curves = {k:[] for k in self.curves}

    def __call__(self):
        self.reset_all(); return self

    def save_json(self, path):
        tmp = self.metrics; del self.metrics
        JSONSerializable.save_json(self, path)
        self.metrics = tmp