from ..utils import *

class Timer(Serializable):
    module_name = "timer"
    module_abbr = "tim"
    def __init__(self, keys=["base"], start=False, logger=Logger(), decode_format="{1} {0}: {2}", time_format="[%Y-%m-%d_%H.%M.%S]", elapsed_format="Elapsed: {0}: {1}"):
        self.decode_format = decode_format; self.time_format = time_format; self.elapsed_format = elapsed_format
        self.data = {}; self.add(keys); self.logger = logger
        if start:
            self.start(keys)
        
    def timing(self, key):
        return self.data[key]['timing']

    def elapsed(self, key):
        current_time = int(time.time_ns()); return self.data[key]['elapsed'] + (current_time-self.data[key]['ticks'][-1][0] if self.timing(key) else 0)

    def add(self, keys):
        _keys = keys if isinstance(keys, list) else [keys]
        for key in _keys:
            self.data.update({
                key:{
                    'timing': False,
                    'ticks': [],
                    'elapsed': 0,
                }
            })

    def start(self, keys):
        _keys = [keys] if isinstance(keys, str) else keys
        for key in _keys:
            if not self.timing(key):
                self.data[key]['timing'] = True
                T = (int(time.time_ns()),"[start timing]")
                self.data[key]['ticks'].append(T)
                self.logger.log(Timer.decode((key, T[0], T[1]),decode_format=self.decode_format,time_format=self.time_format))

    def start_all(self):
        self.start(self.data.keys())

    def pause(self, keys):
        _keys = [keys] if isinstance(keys, str) else keys
        for key in _keys:
            if self.timing(key):
                self.data[key]['timing'] = False
                current_time = int(time.time_ns())
                self.data[key]['elapsed'] += current_time-self.data[key]['ticks'][-1][0]
                T = (current_time,"[pause timing]")
                self.data[key]['ticks'].append(T)
                self.logger.log(Timer.decode((key, T[0], T[1]),decode_format=self.decode_format,time_format=self.time_format))

    def pause_all(self):
        self.pause(self.data.keys())

    def reset(self, keys):
        _keys = [keys] if isinstance(keys, str) else keys
        for key in _keys:
            if self.timing(key):
                self.pause(key)
            self.add(key)
    
    def reset_all(self):
        self.reset(self.data.keys())

    def tick(self, keys, msgs=None):
        if msgs is None:
            _keys = [keys] if isinstance(keys, str) else keys
            _msgs = ["" for _ in range(_keys)]
        else:
            _keys, _msgs = ([keys], [msgs]) if isinstance(keys, str) else (keys, msgs)
            assert (len(_keys)==len(_msgs))
        
        for key,msg in zip(_keys,_msgs):
            current_time = int(time.time_ns())
            if self.timing(key):
                self.data[key]['elapsed'] += current_time-self.data[key]['ticks'][-1][0]
                T = (current_time,"[tick]%s"%msg)
                self.data[key]['ticks'].append(T)
                self.logger.log(Timer.decode((key, T[0], T[1]),decode_format=self.decode_format,time_format=self.time_format))
    
    def tick_all(self):
        self.tick(self.data.keys())
    
    def records(self, keys):
        _keys = [keys] if isinstance(keys, str) else keys
        r = [(key,t,m) for key in _keys for (t,m) in self.data[key]['ticks']]
        r = sorted(r, key=lambda x:x[1]); return r

    def records_all(self):
        return self.records(self.data.keys())

    def decode(record, decode_format="{1} {0}: {2}", time_format="[%Y-%m-%d_%H.%M.%S]"):
        return decode_format.format(record[0],time.strftime(time_format,time.localtime(record[1]/1e9)),record[2])

    def log(self, keys, level=Logger.MAX):
        for r in self.records(keys):
            self.logger.log(Timer.decode(r,decode_format=self.decode_format,time_format=self.time_format),level=level)
        for key in keys:
            t = self.elapsed(key); ms = int(t/1e6); s = int(ms/1e3)
            S = "{:02d}h.{:02d}m.{:02d}s.{:03d}ms.".format(s//3600,s%3600//60,s%60,ms%1000)
            self.logger.log(self.elapsed_format.format(key,S),level=level)

    def log_all(self, level=Logger.MAX):
        return self.log(self.data.keys(),level=level)

    # def __enter__(self):
    #     self.start_time = time.time(); self.int_start_time = int(self.start_time)
    #     print(self.COLOR("[%s START]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(self.int_start_time)))); return self
    # 
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.end_time = time.time(); self.int_end_time = int(self.end_time); interval = self.end_time - self.start_time
    #     print(self.COLOR("[%s   END]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(  self.int_end_time))))
    #     print(self.COLOR("[%s TOTAL]: {:02d}h.{:02d}m.{:02d}s.{:03d}ms.".format(int(interval)//3600,int(interval)%3600//60,int(interval)%60,int(interval*1000)-int(interval)*1000)%self.name))
    # 
    # def tick(self, desc=None):
    #     cur_time = time.time(); int_cur_time = int(cur_time); interval = cur_time - self.start_time
    #     print(self.COLOR("[%s  TICK]"%self.name),self.COLOR(time.strftime("%Y-%m-%d %H:%M:%S.",time.localtime(       cur_time))),self.COLOR("" if desc is None else "(%s)"%desc))
    #     print(self.COLOR("[%s  TILL]: {:02d}h.{:02d}m.{:02d}s.{:03d}ms.".format(int(interval)//3600,int(interval)%3600//60,int(interval)%60,int(interval*1000)-int(interval)*1000)%self.name))
