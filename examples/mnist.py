from matplotlib.pyplot import plot
from mlhp.utils import *
from mlhp.import_all import *
import mlhp.logger
import mlhp.plotter
import mlhp.tracker
import mlhp.common_models as MD
import mlhp.model_wrapper as MR

import torchvision
import torchvision.datasets as TVD
import torchvision.transforms as TVT

def MNIST(partition=0.8, batch_size=32):
    data_train = TVD.MNIST(
        'data/',
        train = True,
        download = True,
        transform = TVT.Compose(
            [TVT.ToTensor(),TVT.Normalize(mean=[0.1307],std=[0.3081])]
        ),
    )
    data_train, data_val = TUD.random_split(data_train, [int(len(data_train)*partition),len(data_train)-int(len(data_train)*partition)])
    data_test = TVD.MNIST(
        'data/',
        train = False,
        download = True,
        transform = TVT.Compose(
            [TVT.ToTensor(),TVT.Normalize(mean=[0.1307],std=[0.3081])]
        ),
    )
    return {
        'train':DataLoader(data_train,batch_size=batch_size,shuffle=True),
        'valid':DataLoader(data_val  ,batch_size=batch_size,shuffle=True),
        'testi':DataLoader(data_test ,batch_size=batch_size,shuffle=False),
    }

class MNISTModelWrapper(MR.VanillaSupervisedModelWrapper):
    def batch_stats(self, pred, true):
        return {'acc':torch.eq(pred.argmax(dim=-1), true).float().mean()}
    
if __name__=="__main__":
    # Loadersß
    loaders = MNIST()
    
    # Model
    net = MD.MLP(
        layer_sizes=[784,512,512,10],
        bias=True,
        activations=[nn.ReLU(inplace=True),nn.ReLU(inplace=True),nn.Identity()],
    )

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr = 0.001,
    )

    # Model Wrapper
    mr = MNISTModelWrapper(
        net=net,
        loss_criterion=nn.CrossEntropyLoss(),
        exp_root="runs",
        exp_name="MNIST",
        exp_info="mlp",
        optimizer=optimizer,
        scheduler=None,
        tracker=None,
        primary="acc",
        epoch=0,
        ddp=False,
        device='cpu',
        reset=True,
        immediate_save=False
    )

    # Logger
    loggers = {
        'train': mlhp.logger.Logger([
            mlhp.logger.Handle(handle="w:"+mr.path+"logs/train.log",priority=mlhp.logger.Logger.DEBUG),
            mlhp.logger.Handle(handle="sys:out",priority=mlhp.logger.Logger.INFO),
        ]),
        'valid': mlhp.logger.Logger([
            mlhp.logger.Handle(handle="w:"+mr.path+"logs/valid.log",priority=mlhp.logger.Logger.INFO),
            mlhp.logger.Handle(handle="sys:out",priority=mlhp.logger.Logger.INFO),
        ]),
        'testi': mlhp.logger.Logger([
            mlhp.logger.Handle(handle="w:"+mr.path+"logs/testi.log",priority=mlhp.logger.Logger.FATAL),
            mlhp.logger.Handle(handle="sys:out",priority=mlhp.logger.Logger.INFO),
        ]),
    }

    # Plotter
    plotter = mlhp.plotter.Plotter(
        figsize=(16,9),
        dpi=300,
        mode="save",
    )

    # Tracker
    tracker = mlhp.tracker.Tracker(
        title="mlp MNIST",
        path=mr.path+"tracker.data",
        metrics=[
            mlhp.tracker.Metric(name="acc",key=lambda x:-x),
            mlhp.tracker.Metric(name="loss",key=lambda x:x),
        ],
        plotter=plotter,
    )
    mr.set_tracker(tracker)

    # Load Config
    config = LoadJSON("mnist_config.json")
    for i,task in enumerate(loaders):
        config['tasks'][i]['loader'] = loaders[task]
    config['__global_args__']['logger'] = mlhp.logger.NONE_LOGGER
    for cfg in config['tasks']:
        cfg['args']['logger'] = loggers[cfg['task']]

    mr.run(100,config)