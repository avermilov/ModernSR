import ast
import configparser
import os.path


class ConfigStruct:
    def __init__(self, **entries):
        converted_entries = self._convert_values(entries)
        self.__dict__.update(converted_entries)

    @staticmethod
    def _boolify(s):
        if s == "True":
            return True
        elif s == "False":
            return False
        raise ValueError("Unknown bool type: " + s)

    def _autoconvert(self, s):
        for fn in (self._boolify, int, float):
            try:
                return fn(s)
            except ValueError:
                pass
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s

    def _convert_values(self, entries):
        for key in entries.keys():
            entries[key] = self._autoconvert(entries[key])
        return entries


class Config:
    def __init__(self, path: str):
        self.path = path
        self._cfg = self._load_cfg()
        self._set_attr()

    def _load_cfg(self):
        cfg = configparser.ConfigParser()
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                cfg.read_file(f)

        return cfg

    def _set_attr(self):
        params_dict = {}
        for item in self._cfg.sections():
            options_dict = dict(self._cfg.items(item))
            params_dict[item.lower()] = ConfigStruct(**options_dict)
        self.__dict__.update(**params_dict)
