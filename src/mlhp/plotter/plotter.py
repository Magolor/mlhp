from ..utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt

class Plotter(JSONSerializable):
    module_name = "plotter"
    module_abbr = "plt"

    supported_modes = ['save','show','none']
    
    def __init__(self, figsize=(8,6), dpi=300, mode="save"):
        self.figsize = figsize; self.dpi = dpi; self.mode = mode
        assert (mode in self.supported_modes), "Supported modes: {0}".format(self.supported_modes)
    
    def __call__(self, title=None, path=None):
        self.title = title; self.path = path; plt.close(); return self

    def __enter__(self):
        fig,axe = plt.subplots(figsize=self.figsize,dpi=self.dpi); axe.set_title(self.title); return (fig,axe)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode=='save':
            plt.savefig(self.path); plt.close()
        elif self.mode=='show':
            plt.show(); plt.close()
        elif self.mode=='none':
            pass

NONE_PLOTTER = Plotter(mode="none")
SHOW_PLOTTER = Plotter(mode="show")
SAVE_PLOTTER = Plotter(mode="save")