from .model_wrapper import *

class VanillaSupervisedModelWrapper(ModelWrapper):
    module_name = "vanilla_supervised_model_wrapper"
    module_abbr = "sup"

    def __init__(self,
        loss_criterion = None,
        **args
    ):
        super(VanillaSupervisedModelWrapper, self).__init__(**args)
        self.loss_criterion = loss_criterion

    # Specify this for each instance
    def batch_stats(self, pred, true):
        raise NotImplementedError

    # Specify this for each instance
    def batch_reduce(self, res):
        return {
            'stats': {k:np.array([r[k] for r in res['stats']]).mean().item() for k in res['stats'][0]} if len(res['stats'])>0 else {},
            'outputs': torch.cat(res['outputs'], dim=0) if len(res['outputs'])>0 else None,
        }

    def run_batch(self, data, return_stats=False, return_outputs=False):
        source, true = data; true = true.cpu(); pred = self(source).cpu()
        res = {'loss':self.loss_criterion(pred, true), 'stats':None, 'outputs':None}
        if return_stats:
            res['stats'] = self.batch_stats(pred, true)
            res['stats']['loss'] = res['loss'].item()
        if return_outputs:
            res['outputs'] = true.detach().cpu()
        return res

    def train_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        assert (self.optimizer is not None), "Optimizer should not be None for training."
        self.train(); res = {'stats':[], 'outputs':[]}
        
        # ddp set_epoch
        if hasattr(loader.sampler,"set_epoch"):
            loader.sampler.set_epoch(self.epoch)
        
        # clear grad
        self.optimizer.zero_grad()
        
        with torch.enable_grad():
            num_batches = len(loader)
            gradient_step = args['gradient_step'] if 'gradient_step' in args else 1
            pbar = TQDM(loader,use_tqdm=args['use_tqdm'])
            for batch_id, data in enumerate(pbar):
                # train batch
                batch_res = self.run_batch(data, return_stats=return_stats, return_outputs=return_outputs)
                batch_res['loss'].backward()
                if (batch_id+1)%gradient_step==0 or (batch_id+1)==num_batches:
                    self.optimizer.step(); self.optimizer.zero_grad()
                
                # stats, outputs
                if return_stats:
                    res['stats'].append(batch_res['stats'])
                if return_outputs:
                    res['outputs'].append(batch_res['outputs'])
                
                # ouput
                output = "[%s]"%self.net.module_name+" Epoch %s Train (loss=%7.4f)"%("#%04d"%self.epoch,batch_res['loss'].item())
                args['logger'].log(output,level=Logger.DEBUG)
                if args['use_tqdm']:
                    pbar.set_description(output,refresh=True)
        
        return self.batch_reduce(res)
        
    def valid_epoch(self, loader, return_stats=False, return_outputs=False, **args):
        if (not args['__final__']) and (args['epoch_per_valid']<=0 or self.epoch%args['epoch_per_valid']!=0):
            return None

        self.eval(); res = {'stats':[], 'outputs':[]}
        
        # ddp set_epoch
        if hasattr(loader.sampler,"set_epoch"):
            loader.sampler.set_epoch(self.epoch)
        
        with torch.no_grad():
            pbar = TQDM(loader,use_tqdm=args['use_tqdm'])
            for batch_id, data in enumerate(pbar):
                # valid batch
                batch_res = self.run_batch(data, return_stats=return_stats, return_outputs=return_outputs)
                
                # stats, outputs\
                if return_stats:
                    res['stats'].append(batch_res['stats'])
                if return_outputs:
                    res['outputs'].append(batch_res['outputs'])
                
                # ouput
                output = "[%s]"%self.net.module_name+" Epoch %s Valid (loss=%7.4f)"%("#%04d"%self.epoch,batch_res['loss'].item())
                args['logger'].log(output,level=Logger.DEBUG)
                if args['use_tqdm']:
                    pbar.set_description(output,refresh=True)
        
        return self.batch_reduce(res)
        