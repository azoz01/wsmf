from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


class GenericRepeatableDataLoader:

    def __init__(self, loader: Iterable[Any]):
        self.batches = list(loader)
        self.released_batches_count = 0

    def __next__(self) -> Any:
        if self.released_batches_count == len(self.batches):
            raise StopIteration()
        batch = self.batches[self.released_batches_count]
        self.released_batches_count += 1
        return batch

    def __iter__(self) -> GenericRepeatableDataLoader:
        return deepcopy(self)

    def __len__(self) -> int:
        return len(self.batches)
