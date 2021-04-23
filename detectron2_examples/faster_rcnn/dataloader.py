#!/usr/bin/env python

from torch.utils.data import Dataset
from torchvision import transforms as T


class Dataloader(Dataset):
    def __init__(self, data_source, transforms: list = None, **kwargs: dict):
        super(Dataloader, self).__init__()

        if transforms is not None:
            assert not isinstance(transforms, T.Compose)
        self.data_source = data_source
        self.transforms = transforms

    def __getitem__(self, index: int):
        data = self.data_source(index)
        return data if self.transforms is None else self.transforms(data)

    def __len__(self):
        return len(self.data_source)


if __name__ == '__main__':

    import sys
    from car196 import Car196

    path = sys.argv[1]
    c = Car196(root=path)
    d = Dataloader(data_source=c).__getitem__(20)
    print(d)
