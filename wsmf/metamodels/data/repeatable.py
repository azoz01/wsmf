from copy import deepcopy
from typing import Iterable


class GenericRepeatableD2vLoader:

    def __init__(self, loader: Iterable):
        self.batches = list(loader)
        self.released_batches_count = 0

    def __next__(self):
        if self.released_batches_count == len(self.batches):
            raise StopIteration()
        batch = self.batches[self.released_batches_count]
        self.released_batches_count += 1
        return batch

    def __iter__(self):
        return deepcopy(self)

    def __len__(self):

        return len(self.batches)
