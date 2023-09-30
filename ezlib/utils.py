from PIL.ExifTags import TAGS

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

