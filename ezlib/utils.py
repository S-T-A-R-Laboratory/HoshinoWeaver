import time
from typing import Union, Callable
from logging import getLogger
import dataclasses
import numpy as np
from typing import Union

from PIL.ExifTags import TAGS

logger = getLogger()

class MetaInfo(object):
    """ 

    # MetaInfo

    MetaInfo is used to store metadata. In MetaInfo, tags can be indexed and 
    corresponding values can be got as using string values of tags.

    ## Args:
        init_dict (dict): Metadata dictionary.
        encoding (str, optional): indicates how to decode byte array in metadata. 
                    Defaults to "latin-1".
        str_maxlen (int, optional): the max length for formatting and print, 
                    string longer than which will be shown as "<LongString>". Defaults to 20.
        
    ## Usage
    
    > Initialization. To initialize, simply run `MetaObj=MetaInfo(meta_dict)`, 
    where meta_dict is metadata dictionary.

    > Get available attributes. Since `MetaInfo` is iterable, you can get all 
    available attributes by running `[for tags in MetaObj]`.

    > Get attributes. There are several ways to get a specific attribute. If 
    you want to get "icc_profile", you can use `MetaObj.icc_profile` or 
    `MetaObj.get("icc_profile")`. If you like, 
    `MetaObj.main_dict[MetaObj.tags["icc_profile"]]` can also help you...perhaps.

    > MetaInfo also provides a graceful print list (not exactly). 
    If you want to have a try, run `print(MetaObj)`.

    """

    def __init__(self, init_dict, encoding="latin-1", str_maxlen=20):
        self.main_dict = init_dict
        self.tags = {
            TAGS.get(tag_id, tag_id): tag_id
            for tag_id in self.main_dict
        }
        for tag, tag_id in self.tags.items():
            self.__setattr__(tag, self.main_dict[tag_id])

        self.encoding = encoding
        self.str_maxlen = str_maxlen

    def get(self, string, default=None):
        if string in self.tags:
            return self.main_dict[self.tags[string]]
        if string in self.main_dict:
            return self.main_dict[string]
        return default

    def fmt(self, string):
        string = string.decode(self.encoding) if isinstance(
            string, bytes) else str(string)
        if len(string) > self.str_maxlen:
            return "<LongString>"
        return string

    def __repr__(self) -> str:
        ret_str = "\nMetaInfoList:\n"
        for (key, item) in self.tags.items():
            ret_str += f"{key:<20s}: {self.fmt(self.main_dict[item])} \n"
        return ret_str

    def __iter__(self):
        for tag in self.tags:
            yield tag


def get_resize(tgt_wh: int, raw_wh: Union[list, tuple]):
    w, h = raw_wh
    tgt_wh = [tgt_wh, -1] if w > h else [-1, tgt_wh]
    idn = 0 if tgt_wh[0] <= 0 else 1
    idx = 1 - idn
    tgt_wh[idn] = int(raw_wh[idn] * tgt_wh[idx] / raw_wh[idx])
    return tgt_wh

def time_cost_warpper(func: Callable)->Callable:
    def do_func(*args,**kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        logger.info(f"{func.__name__} Time Cost: {(time.time()-t0):.2f}s.")
        return res
    return do_func


@dataclasses.dataclass
class GaussianParam(object):
    mu: Union[np.ndarray,int,float]
    var: Union[np.ndarray,int,float] = 0
    n: int = 1
    ddof: int = 1

    def __add__(g1, g2):
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n + g2.mu * g2.n) / (g1.n + g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) +
                   (g2.n - ddof) * g2.var + g2.n * np.square(g2.mu) -
                   (g1.n + g2.n) * np.square(new_mu)) / (g1.n + g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n + g2.n, ddof=ddof)

    def __sub__(g1, g2):
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n>g2.n, "generate n<0 fistribution!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n - g2.mu * g2.n) / (g1.n - g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) -
                   (g2.n - ddof) * g2.var - g2.n * np.square(g2.mu) -
                   (g1.n - g2.n) * np.square(new_mu)) / (g1.n - g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n - g2.n, ddof=ddof)


def test_GaussianParam():
    tot_num = 40
    series = np.random.randn(tot_num)
    base = GaussianParam(mu=series[0])
    for i in range(1, tot_num):
        print("n =", i)
        add = GaussianParam(mu=series[i])
        base = base + add
        print(base.mu,np.mean(series[:i + 1]))
        assert np.abs(base.mu - np.mean(series[:i + 1])) < 1e-8, (
            base.mu, np.mean(series[:i + 1]))
        assert np.abs(base.var - np.var(series[:i + 1], ddof=1)) < 1e-8, (
            base.var, np.var(series[:i + 1], ddof=1), series)
        assert base.n == i + 1
        assert base.ddof == 1
