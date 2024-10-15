import torch
from torch.utils.data import DataLoader
from random import choice
from bisect import bisect_left
from typing import Any, List, Callable, Optional, Iterable


class DataIterator:
    def __init__(self, loader, auto_reset=True):
        self.loader = loader
        self.iter = iter(self.loader)
        self.auto_reset = auto_reset

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            if self.auto_reset:
                self.reset()
                return next(self.iter)
            else:
                return

    def reset(self):
        self.iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)


class CUDAPreFetchDataIterator:
    def __init__(self, loader, auto_reset=True):
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.preload()
        self.auto_reset = auto_reset

    def preload(self):
        try:
            one_batch = next(self.iter)
        except StopIteration:
            if self.auto_reset:
                self.reset()
                one_batch = next(self.iter)
            else:
                return
        with torch.cuda.stream(self.stream):
            if isinstance(one_batch, torch.Tensor):
                self.next_batch = one_batch.cuda(non_blocking=True)
            else:
                next_batch = []
                for item in one_batch:
                    if isinstance(item, torch.Tensor):
                        next_batch.append(item.cuda(non_blocking=True))
                    elif isinstance(item, list) or isinstance(item, tuple):
                        next_batch.append([t.cuda(non_blocking=True) for t in item])
                self.next_batch = next_batch

    def reset(self):
        self.iter = iter(self.loader)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        current_batch = self.next_batch
        self.preload()
        return current_batch

    def __len__(self):
        return len(self.loader)


class DataLoaderManager:
    def __init__(
        self,
        loaders:List[DataLoader],
        milestones:Optional[Iterable]=None,
        progressive:bool=False,
        cuda_prefetch:bool=False,
        hook_func:Callable=None,
    ) -> None:
        IteratorClass = CUDAPreFetchDataIterator if cuda_prefetch else DataIterator
        iterators = [IteratorClass(loader) for loader in loaders]
        if milestones is not None:
            if len(milestones) == len(iterators) - 1:
                intervals = [0.0] + milestones
            else:
                intervals = milestones
            assert len(intervals) == len(iterators)
        elif progressive:
            intervals = [i / len(iterators) for i in range(len(iterators))]
        else:
            intervals = None
        self.iterators = iterators
        self.intervals = intervals
        self.hook_func = hook_func
    
    def __call__(self, train_step:int, max_step:int) -> Any:
        if self.intervals is None:
            iterator = choice(self.iterators)
        else:
            iterator_index = bisect_left(self.intervals, train_step / max_step) - 1
            iterator = self.iterators[iterator_index]
        if self.hook_func is not None:
            self.hook_func(iterator, train_step, max_step)
        return iterator.next()


def get_data_iterator(loader, cuda_prefetch=False, verbose=False):
    assert type(loader) is DataLoader
    if cuda_prefetch:
        if verbose:
            print('use CUDA-prefetch data iterator!')
        return CUDAPreFetchDataIterator(loader)
    else:
        return DataIterator(loader)


if __name__ == '__main__':
    data_iter = get_data_iterator(DataLoader(...))
    img, label = data_iter.next()

    while img is not None:
        # processing...
        img, label = data_iter.next()

