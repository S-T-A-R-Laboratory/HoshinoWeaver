"""An explicit, revocable destination for a saved mask path."""
from dataclasses import dataclass
from typing import Callable


@dataclass(eq=False)
class MaskTarget:
    label: str
    path: str
    is_current: Callable[[], bool]
    set_path: Callable[[str], None]
    read_path: Callable[[], str] | None = None

    @property
    def current_path(self):
        return self.read_path() if self.read_path else self.path

    def apply(self, path):
        if not self.is_current():
            return False
        self.set_path(path)
        self.path = path
        return True
