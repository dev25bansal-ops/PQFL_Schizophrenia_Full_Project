"""Federated learning module for PQFL.

Implements personalized quantum federated learning using Flower,
with FedPer/FedProx strategies and quantum-specific parameter management.
"""

from .client import PQFLClient
from .strategy import PQFLStrategy, PQFLFedPerStrategy, PQFLFedProxStrategy
from .server import PQFLServer
from .parameter_utils import (
    get_shared_parameters,
    set_shared_parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

__all__ = [
    "PQFLClient",
    "PQFLStrategy",
    "PQFLFedPerStrategy",
    "PQFLFedProxStrategy",
    "PQFLServer",
    "get_shared_parameters",
    "set_shared_parameters",
    "parameters_to_ndarrays",
    "ndarrays_to_parameters",
]
