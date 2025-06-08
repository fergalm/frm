

from abc import ABC, abstractmethod
from typing import Hashable
import fsspec
import os

class AbstractKeyVault(ABC):
    @abstractmethod
    def get(self, key:Hashable, default=None) -> str:
        """Return the secret associated with `key`.

        Raise a `KeyError` if not found.
        """
        pass

    def __getitem__(self, key:Hashable):
        return self.get(key)

    def getdefault(self, key:Hashable, default):
        """Return the secret associated with `key`.

        Return the `default` value if key not found.
        """
        try:
            self.get(key)
        except KeyError:
            return default

    @abstractmethod
    def list(self):
        pass


class EnvKeyVault(AbstractKeyVault):
    def get(self, key:Hashable):
        return os.environ[key]

    def list(self):
        return list(os.environ.keys())

class TomlVault(AbstractKeyVault):
    def __init__(self, path, fs=None):
        fs = fs or fsspec.filesystem('file')
        with fs.open(path) as fp:
            self.vault = toml.load(fp)

    def get(self, key:Hashable):
        return self.vault[key]

    def list(self):
        return list(self.vault.keys())


