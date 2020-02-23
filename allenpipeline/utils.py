from typing import Dict, Any, List, Tuple, Iterable


def flatten(d : Dict[Any, Any]):
    """
    Flattens a dictionary and uses the path separated with _ to give unique key names.
    :param d:
    :return:
    """
    r = dict()
    agenda : List[Tuple[Any, List, Any]] = [ (key,[],d) for key in d.keys()]
    while agenda:
        key,path,d = agenda.pop()
        if not isinstance(d[key],dict):
            r["_".join(path+[str(key)])] = d[key]
        else:
            for subkey in d[key].keys():
                agenda.append((subkey,path+[str(key)],d[key]))
    return r

def merge_dicts(x: Dict, prefix:str, y: Dict):
    r = dict()
    for k,v in x.items():
        r[k] = v
    for k,v in y.items():
        r[prefix+"_"+k] = v
    return r


def get_hyperparams(d : Dict) -> Iterable[Tuple]:
    agenda = [("",d)]
    while agenda:
        name, d = agenda.pop()
        if name == "":
            prefix = ""
        else:
            prefix = name + ":"
        for key, val in d.items():
            if isinstance(val,str) or isinstance(val, int) or isinstance(val, float):
                yield prefix+key,val
            elif isinstance(val, dict):
                agenda.append((prefix+key,val))
