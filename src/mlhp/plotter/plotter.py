from ..utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt

class Plotter(Serializable):
    module_name = "plotter"
    module_abbr = "plt"

    supported_modes = ['save','show']
    
    def __init__(self, title=None, path=None, figsize=(16,9), mode="save"):
        self.title = title; self.path = path; self.figsize = figsize; self.mode = mode
        assert (mode in self.supported_modes), "Supported modes: {0}".format(self.supported_modes)
    
    def __enter__(self):
        fig,axe = plt.subplots(figsize=self.figsize,dpi=300); axe.set_title(self.title); return (fig,axe)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode=='save':
            plt.savefig(self.path); plt.close()
        elif self.mode=='show':
            plt.show(); plt.close()