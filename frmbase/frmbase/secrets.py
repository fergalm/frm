
from abc import ABC, abstractmethod
import fsspec
import os

class AbstractKeyVault(ABC):
    @abstractmethod
    def get(self, key) -> str:
        pass

    def __getitem__(self, key):
        return self.get(key)

    @abstractmethod
    def list(self):
        pass


class EnvKeyVault(AbstractKeyVault):
    def get(self, key):
        value = os.environ[key]
        return value

    def list(self):
        return list(os.environ.keys())

class TomlVault(AbstractKeyVault):
    def __init__(self, path, fs=None):
        fs = fs or fsspec.filesystem('file')
        with fs.open(path) as fp:
            self.vault = toml.load(fp)

    def get(self, key):
        return self.vault[key]

    def list(self):
        return list(self.vault.keys())


