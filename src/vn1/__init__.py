"""One arm per module, loaded lazily like `vn2.forecaster`."""
from .forecaster import chronos, ma12, naive_last, tabpfn_api

__all__ = ["chronos", "ma12", "naive_last", "tabpfn_api"]
