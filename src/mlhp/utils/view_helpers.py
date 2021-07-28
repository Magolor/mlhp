from ..import_basic import *
import tqdm

def ViewSobj(obj, length=4096, charcount=False):
    s = str(obj); return s[:length]+(" ...[omitting %d chars]"%len(s[length:]) if charcount else " ...") if len(s)>length+3 else s

def Viewobj(obj, length=4096, charcount=False):
    print(ViewSobj(obj, length=length, charcount=charcount))

def ViewSList(obj, length=4096, charcount=False, items=256, itemcount=False, last_comma=True, bracket='[]'):
    S = "{0}\n".format(bracket[0])
    for i,value in enumerate(list(obj)[:items]):
        S += "\t" + ViewSobj(value,length=length,charcount=charcount) + (",\n" if last_comma or i<len(obj)-1 else "\n")
    if len(obj) > items:
        S += "\t...[omitting %d items]\n"%(len(obj)-items) if itemcount else "\t...\n"
    S += "{0}\n".format(bracket[1])
    return S

def ViewList(obj, length=4096, charcount=False, items=256, itemcount=False, last_comma=True, bracket='[]'):
    print(ViewSList(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma, bracket=bracket))


def ViewSDict(obj, length=4096, charcount=False, items=256, itemcount=False, last_comma=True):
    S = "{\n"
    for i,(key,value) in enumerate(list(obj.items())[:items]):
        S += "\t" + str(key) + ": " + ViewSobj(value,length=length,charcount=charcount) + (",\n" if last_comma or i<len(obj)-1 else "\n")
    if len(obj) > items:
        S += "\t...[omitting %d items]\n"%(len(obj)-items) if itemcount else "\t...\n"
    S += "}\n"
    return S

def ViewDict(obj, length=4096, charcount=False, items=256, itemcount=False, last_comma=True):
    print(ViewSDict(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma))


def ViewSJSON(obj, length=4096, charcount=False, indent=4):
    return ViewSobj(json.dumps(obj,indent=indent),length=length,charcount=charcount)

def ViewJSON(obj, length=4096, charcount=False, indent=4):
    print(ViewSJSON(obj, length=length, charcount=charcount, indent=indent))


def ViewS(obj, length=4096, charcount=False, items=None, itemcount=None, last_comma=None, mode='deduce', indent=None, bracket='[]'):
    modes = ['deduce','dict','list','json','obj']
    assert (mode in modes), "Supported Modes: {0}".format(modes)
    if mode=='deduce':
        if isinstance(obj, list):
            return ViewS(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma, mode="list", indent=indent, bracket=bracket)
        for m in modes[1:]:
            try:
                return ViewS(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma, mode=m, indent=indent, bracket=bracket)
            except:
                pass
        assert False, "Failed to deduce"
    elif mode=='dict':
        assert ((indent is None) and (bracket is None))
        items = 256 if items is None else items
        itemcount = False if itemcount is None else itemcount
        last_comma = True if last_comma is None else last_comma
        return ViewSDict(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma)
    elif mode=='list':
        assert ((indent is None))
        items = 256 if items is None else items
        itemcount = False if itemcount is None else itemcount
        last_comma = True if last_comma is None else last_comma
        return ViewSList(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma, bracket=bracket)
    elif mode=='json':
        assert ((items is None) and (itemcount is None) and (last_comma is None))
        return ViewSJSON(obj, length=length, charcount=charcount, indent=indent)
    elif mode == 'obj':
        assert ((items is None) and (itemcount is None) and (last_comma is None) and (indent is None))
        return ViewSobj(obj, length=length, charcount=charcount)
    else:
        raise NotImplementedError
    
def View(obj, length=4096, charcount=False, items=None, itemcount=None, last_comma=None, mode='deduce', indent=None, bracket='[]'):
    print(ViewS(obj, length=length, charcount=charcount, items=items, itemcount=itemcount, last_comma=last_comma, mode=mode, indent=indent, bracket=bracket))


def PrintConsole(*args, **kwargs):
    print(*args, file=sys.stdout, flush=True, **kwargs)

def PrintError(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def NORMAL(obj):
    return str(obj)

def ERROR(obj):
    return "\033[1;31m"+str(obj)+"\033[0m"

def SUCCESS(obj):
    return "\033[1;32m"+str(obj)+"\033[0m"

def WARNING(obj):
    return "\033[1;33m"+str(obj)+"\033[0m"

def COLOR1(obj):
    return "\033[1;34m"+str(obj)+"\033[0m"

def COLOR2(obj):
    return "\033[1;35m"+str(obj)+"\033[0m"

def COLOR3(obj):
    return "\033[1;36m"+str(obj)+"\033[0m"

def BLACK(obj):
    return "\033[1;30m"+str(obj)+"\033[0m"

def RED(obj):
    return "\033[1;31m"+str(obj)+"\033[0m"

def GREEN(obj):
    return "\033[1;32m"+str(obj)+"\033[0m"

def YELLOW(obj):
    return "\033[1;33m"+str(obj)+"\033[0m"

def BLUE(obj):
    return "\033[1;34m"+str(obj)+"\033[0m"

def MAGENTA(obj):
    return "\033[1;35m"+str(obj)+"\033[0m"

def CYAN(obj):
    return "\033[1;36m"+str(obj)+"\033[0m"

def WHITE(obj):
    return "\033[1;37m"+str(obj)+"\033[0m"


def TQDM(obj, s=0, desc=None, use_tqdm=True, **kwargs):
    if use_tqdm:
        if type(obj) is int:
            return tqdm.trange(s,obj+s,total=obj,desc=desc,dynamic_ncols=True, **kwargs)
        else:
            assert (s==0)
            return tqdm.tqdm(obj,total=len(obj),desc=desc,dynamic_ncols=True, **kwargs)
    else:
        if type(obj) is int:
            return range(s,obj+s)
        else:
            assert (s==0)
            return obj