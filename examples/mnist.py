from mlhp.ml_usage import *
from mlhp.import_vision import *

# Define dataset
def MNISTLoaders(partition=0.8, batch_size=32):
    data_train = TVD.MNIST('data/', train = True, download = True,
        transform = TVT.Compose([TVT.ToTensor(),TVT.Normalize(mean=[0.1307],std=[0.3081])]),
    )
    data_train, data_val = TUD.random_split(data_train, 
        [int(len(data_train)*partition),len(data_train)-int(len(data_train)*partition)]
    )
    data_test = TVD.MNIST('data/', train = False, download = True,
        transform = TVT.Compose([TVT.ToTensor(),TVT.Normalize(mean=[0.1307],std=[0.3081])]),
    )
    return [
        DataLoader(data_train,batch_size=batch_size,shuffle=True),
        DataLoader(data_val  ,batch_size=batch_size,shuffle=True),
        DataLoader(data_test ,batch_size=batch_size,shuffle=False),
    ]

# Define a wrapper easily by inheriting a supervised learning framework
class MNISTModelWrapper(MR.VanillaSupervisedModelWrapper):
    def batch_stats(self, pred, true):
        return {'acc':torch.eq(pred.argmax(dim=-1), true).float().mean()}
    
if __name__=="__main__":
    
    num_epoch = 2

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

    # Scheduler
    scheduler = sched.CosineAnnealingLR(
        optimizer,
        T_max = num_epoch,
        eta_min = 1e-5,
    )

    # Model Wrapper
    mr, config = MR.setup_vanilla_supervised_model_wrapper(
        # Experiment Args
        exp_root        = "runs",
        exp_name        = "MNIST",
        exp_info        = "mlp",
        subclass        = MNISTModelWrapper,
        loaders         = MNISTLoaders(),
        config_path     = "configs/example_supervised_config.json",

        # Network Args
        net             = net,
        loss_criterion  = nn.CrossEntropyLoss(),
        optimizer       = optimizer,
        scheduler       = scheduler,
        metrics         = [
            tracker.Metric(name='loss',key=lambda x:x),
            tracker.Metric(name='acc',key=lambda x:-x),
        ],
        primary         = "accs",
        
        # Running Args
        start_epoch     = 0,
        ddp             = False,
        device          = "cpu",
        reset           = True,

        # Logging Args
        log_dir         = "logs/",
        plt_figsize     = (16,9),
        plt_dpi         = 300,
        tracker_dir     = "stats/",
    )
    
    ViewDict(config)
    
    # Timer
    T = timer.Timer(keys=["base"])
    T.start_all()

    mr.run(num_epoch, config)

    T.pause_all()