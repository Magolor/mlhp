from ..import_basic import *
import jsonlines
import pickle

def PathToString(path):
    return path if isinstance(path, str) else str(path) + '/'*path.is_dir()

# Short for PathToString
def p2s(path):
    return path if isinstance(path, str) else str(path) + '/'*path.is_dir()

def AbsPath(path):
    return PathToString(Path(path).absolute())


def Folder(path):
    parts = PathToString(path).split('/'); return ('/'.join(parts[:-1])+'/') if len(parts)>1 else './'
    
def File(path):
    return PathToString(path).split('/')[-1]

def Prefix(path):
    parts = PathToString(path).split('.'); return '.'.join(parts[:-1])

def Suffix(path):
    return PathToString(path).split('.')[-1]

def Format(path):
    return PathToString(path).split('.')[-1]

def AddFormat(path, format):
    return PathToString(path) + '.' + format


def Parent(path):
    parts = PathToString(path).split('/')
    if parts[-1]=='':
        parts = parts[-1]
    return ('/'.join(parts[:-1])+'/') if len(parts)>1 else './'
    
def IsFolder(path):
    path = PathToString(path); return path==''  or path[-1]=='/'

def IsFile(path):
    path = PathToString(path); return path!='' and path[-1]!='/'


def Exist(path):
    return Path(path).exists()
    
def ExistFolder(path):
    return Path(path).is_dir()

def ExistFile(path):
    return Path(path).is_file()
    

def ListDir(path, ordered=False, with_path=False, **sort_args):
    objs = sorted(os.listdir(path),**sort_args) if ordered else os.listdir(path)
    return [PathToString(Path(path)/Path(obj)) for obj in objs] if with_path else objs

def ListFolders(path, ordered=False, with_path=False, **sort_args):
    objs = sorted([d for d in os.listdir(path) if ExistFolder(path + d)],**sort_args) if ordered else [d for d in os.listdir(path) if ExistFolder(path + d)]
    return [PathToString(Path(path)/Path(obj)) for obj in objs] if with_path else objs

def ListFiles(path, ordered=False, with_path=False, **sort_args):
    objs = sorted([d for d in os.listdir(path) if ExistFile(path + d)],**sort_args) if ordered else [d for d in os.listdir(path) if ExistFile(path + d)]
    return [PathToString(Path(path)/Path(obj)) for obj in objs] if with_path else objs

def NewestDir(path, with_path=False, topk=1):
    key = (lambda x:os.path.getctime(x)) if with_path else (lambda x:os.path.getctime(path + x))
    return ListDir(path, ordered=True, key=key, with_path=with_path, reverse=True)[:topk]

def NewestFolders(path, with_path=False, topk=1):
    key = (lambda x:os.path.getctime(x)) if with_path else (lambda x:os.path.getctime(path + x))
    return ListFolders(path, ordered=True, key=key, with_path=with_path, reverse=True)[:topk]

def NewestFiles(path, with_path=False, topk=1):
    key = (lambda x:os.path.getctime(x)) if with_path else (lambda x:os.path.getctime(path + x))
    return ListFiles(path, ordered=True, key=key, with_path=with_path, reverse=True)[:topk]


def CreateFolder(path):
    if not Exist(path):
        Path(path).mkdir(parents=True,exist_ok=False); return True
    else:
        return False
    
def CreateFile(path):
    if not Exist(path):
        Path(Folder(path)).mkdir(parents=True,exist_ok=True)
        if File(path)!='':
            Path(path).touch(exist_ok=False)
        return True
    else:
        return False

def Delete(path):
    if Exist(path):
        path = PathToString(path)
        shutil.rmtree(path) if ExistFolder(path) else os.remove(path); return True
    else:
        return False

def ClearFolder(path):
    Delete(path); return CreateFolder(path)

def ClearFile(path):
    Delete(path); return CreateFile(path)


def SaveJSON(obj, path, jsonl=False, indent=None, append=None):
    CreateFile(path); path = PathToString(path)
    if jsonl:
        assert (indent is None), "'jsonl' format does not support parameter 'indent'!"
        with jsonlines.open(path, 'a' if append else 'w') as f:
            for data in obj:
                f.write(data)
    else:
        assert (append is None), "'json' format does not support parameter 'append'!"
        with open(path, 'w') as f:
            json.dump(obj, f, indent=indent)

def LoadJSON(path, jsonl=False):
    assert (ExistFile(path)), "path '{0}' does not exist".format(path)
    path = PathToString(path)
    if jsonl:
        with open(path, 'r') as f:
            return [data for data in jsonlines.Reader(f)]
    else:
        with open(path, 'r') as f:
            return json.load(f)

def PrettifyJSON(path, indent=4):
    assert (ExistFile(path)), "path '{0}' does not exist".format(path)
    SaveJSON(LoadJSON(path),path,indent=indent)


def SavePickle(obj, path, protocol=None):
    CreateFile(path); path = PathToString(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def LoadPickle(path):
    assert (ExistFile(path)), "path '{0}' does not exist".format(path)
    return pickle.load(open(PathToString(path),'rb'))
