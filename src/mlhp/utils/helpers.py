from ..import_basic import *
import requests

def IP():
    return requests.get('https://api.ipify.org').text

def DATETIME(format="[%Y-%m-%d_%H.%M.%S]"):
    return time.strftime(format,time.localtime(time.time()))

def CMD(command, sudo=False, wait=True):
    h = subprocess.Popen(("sudo " if sudo else "")+command,shell=True); return h.wait() if wait else h

def PIP(package, source='', https=False, upgrade=False, force=False, force_deps=False, pip3=True, args=""):
    package = package.lower()   # Pypi is case insensitive
    index = {
        '':"",
        'aliyun':"http://mirrors.aliyun.com/pypi/simple/",
        'douban':"http://pypi.douban.com/simple/",
        'tuna':"http://pypi.tuna.tsinghua.edu.cn/simple/",
        'ustc':"http://pypi.mirrors.ustc.edu.cn/simple/",
    }
    if https:
        index = {k:v.replace("http://","https://") for k,v in index.items()}
    trust = {k:(v.strip("http://").split('/')[0] if k!='' else v) for k,v in index.items()}
    if source not in index:
        print("Source '%s' not found! Use custom."%source)
        index[source] = source
        trust[source] = source
    CMD("{1} install {0} {2}{3}{4}{5}{6}"
        .format(package,
            "pip3" if pip3 else "pip",
            "-i {0} --trusted-host {1} ".format(index[source], trust[source]) if source!='' else "",
            "--upgrade " if upgrade else "",
            "--force-reinstall " if force else "",
            "--no-deps " if (force and not force_deps) else "",
        args)
    )

def CONDA(package, update=False, force=False, force_deps=False, c="", args=""):
    package = package.lower()   # Pypi is case insensitive
    CMD("pip {1} {0} {2}{3}{4}{5}"
        .format(package,
            "update" if update else "install",
            "-c "+c+" " if c!="" else "",
            "--force-reinstall " if force else "",
            "--no-deps " if (force and not force_deps) else "",
        args)
    )


def RandString(length, charset=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(charset) for _ in range(length))

def LineToFloats(line):
    return [float(s) for s in re.findall(r"(?<!\w)[-+]?\d*\.?\d+(?!\d)",line)]

def FlattenList(L):
    F = lambda x:[e for i in x for e in F(i)] if isinstance(x,list) else [x]; return F[L]