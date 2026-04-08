from .. import ops
from ..ops.base import BaseOp

REGISTERED_OP: dict[str, type[BaseOp]] = {}


def register_op(*names: str):

    def decorator(cls):
        for name in names or [cls.__name__]:
            REGISTERED_OP[name] = cls
        return cls

    return decorator
