from ..utils import *
from ..logger import *
from ..tracker import *
from ..import_torch import *
from ..torch_serializable.torch_serializable import TorchSerializable

class ModelWrapper(nn.Module, TorchSerializable):
    module_name = "model_wrapper"
    module_abbr = "wrp"

    def __init__(self,
        net = None,
        exp_root = None,
        exp_name = None,
        exp_info = None,
        optimizer = None,
        scheduler = None,
        tracker = Tracker(),
        primary = None,
        epoch = 0,
        ddp = False,
        ddp_gpu = "0,1,2,3,4,5,6,7",
        device = 'cpu',
        reset = False,
        immediate_save = None,
    ):
        super(ModelWrapper, self).__init__()
        self.net = net
        self.eval()

        self.exp_root = exp_root
        self.exp_name = exp_name
        self.exp_info = exp_info
        if (exp_root is not None) and (exp_name is not None) and (exp_info is not None):
            self.path = Path(exp_root)/Path(exp_name)/Path(exp_info)
            ClearFolder(self.path) if reset else CreateFolder(self.path)
            self.path = PathToString(self.path)
        else:
            self.path = None

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.tracker = tracker
        self.primary = primary
        self.epoch = epoch

        self.ddp = ddp
        
        if ddp:
            os.environ['CUDA_VISIBLE_DEVICES'] = ddp_gpu
            torch.distributed.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        self.to(device)
        if ddp:
            self.net = DDP(self.net,find_unused_parameters=True)

        if immediate_save is not None:
            self.save(path=immediate_save)

    def to(self, device):
        self.device = device
        if self.net is not None:
            self.net.to(device)
        return self
    
    def eval(self):
        if self.net is not None:
            self.net.eval()
        return self

    def train(self):
        if self.net is not None:
            self.net.train()
        return self

    def save(self, path=None):
        if path is None:
            path = "{:s}e{:06d}.pth"
        path = path.format(self.path, self.epoch); tmp = None
        if self.ddp:
            tmp = self.net; self.net = self.net.module
        TorchSerializable.save(self, path)
        if self.ddp:
            self.net = tmp
        return path

    def set_tracker(self, tracker):
        self.tracker = tracker

    def load(self, path=None, device=None):
        if path is None:
            path = "{:s}e{:06d}.pth"
        if ExistFile(path):
            path = path.format(self.path, self.epoch)
            TorchSerializable.load(self, path)
        self.to(self.device if device is None else device)
        if self.ddp:
            self.net = DDP(self.net,find_unused_parameters=True)
        return self

    def forward(self, data):
        return self.net(data)

    # Specify this for each instance
    def run_batch(self, data, return_stats=False, return_outputs=False):
        raise NotImplementedError

    # Specify this for each instance if needed
    def batch_stats(self, pred, data):
        raise NotImplementedError

    # Specify this for each instance if needed
    def batch_reduce(self, res):
        return res

    # Specify this for each instance if needed
    def train_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        raise NotImplementedError

    # Specify this for each instance if needed
    def valid_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        if (not args['__final__']) and (args['epoch_per_valid']<=0 or self.epoch%args['epoch_per_valid']!=0):
            return None
        raise NotImplementedError

    # Specify this for each instance if needed
    def testi_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        return self.valid_epoch(loader, return_stats=return_stats, return_outputs=return_outputs, task="testi", **args)

    # Specify this for each instance if needed
    def devel_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        return self.testi_epoch(loader, return_stats=return_stats, return_outputs=return_outputs, task="devel", **args)

    def run_epoch(self, config, final=True):
        # Check that all tasks are well-defined
        for cfg in config['tasks']:
            task_func = cfg['task']+"_epoch"
            assert (hasattr(self, task_func)), f"'{task_func}' is not defined!"

        res = []; epoch_inc = 0
        for cfg in config['tasks']:
            inc = cfg['pre_increment_epoch'] if 'pre_increment_epoch' in cfg else 0
            self.epoch += inc; epoch_inc += inc
            
            task_func = cfg['task']+"_epoch"
            args = deepcopy(config['__global_args__'])
            args['__final__'] = final
            args.update(cfg['args'])
            res.append(
                getattr(self, task_func)(
                    loader=cfg['loader'],
                    return_stats=cfg['return_stats'],
                    return_outputs=cfg['return_outputs'],
                    **args
                )
            )

            if self.scheduler is not None:
                for _ in range(inc):
                    self.scheduler.step()
        
        post_increment_epoch_to = config['__global_args__']['post_increment_epoch_to'] if 'post_increment_epoch_to' in config['__global_args__'] else 1
        if epoch_inc < post_increment_epoch_to:
            inc = post_increment_epoch_to-epoch_inc
            self.epoch += inc
            if self.scheduler is not None:
                for _ in range(inc):
                    self.scheduler.step()

        epoch_per_save = config['__global_args__']['epoch_per_save'] if 'epoch_per_save' in config['__global_args__'] else 1
        if epoch_per_save <= 0 or self.epoch%epoch_per_save==0:
            self.save(config['__global_args__']['save_path'])

        return res

    # Specify this for each instance if needed
    def run(self, num_iters, config):
        for i in range(num_iters):
            res = self.run_epoch(config,final=(i==num_iters-1))
            for r,t in zip(res[-1],config['tasks']):
                if t['return_stats'] and (r is not None):
                    output = "[%s]"%self.net.module_name+" Epoch %s %s (%s)"%("#%04d"%self.epoch,t['task'],
                             ', '.join([f"{k}={'%.4f'%v}" for k,v in r['stats'].items()]))
                    t['args']['logger'].log(output,level=Logger.FATAL if (i==num_iters-1) else Logger.INFO)
                    if t['task']=='valid':
                        for k,v in r['stats'].items():
                            updated = self.tracker.update(k, self.epoch, v)
                            if updated and k==self.primary:
                                self.save(path=config['__global_args__']['best_path'])
            self.tracker.plot()
            self.tracker.save(self.tracker.path)
            self.tracker.save_json(Prefix(self.tracker.path)+".json")
            yield res