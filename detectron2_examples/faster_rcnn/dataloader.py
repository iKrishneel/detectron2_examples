#!/usr/bin/env python

from torch.utils.data import Dataset
from torchvision import transforms as T


def default_transforms() -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
        ]
    )


class Dataloader(Dataset):
    def __init__(self, data_source, transforms: list = None, **kwargs: dict):
        super(Dataloader, self).__init__()

        if transforms is not None:
            assert not isinstance(transforms, T.Compose)
        self.data_source = data_source
        self.transforms = transforms

        self._is_test = kwargs.get('is_test', False)

    def __getitem__(self, index: int):
        image, target = self.data_source.get(index)
        if self._is_test:
            return default_transforms()(image), target
        return (image, target) if self.transforms is None else self.transforms(
            image
        ), target

    def __len__(self):
        return len(self.data_source)


if __name__ == '__main__':

    import sys
    from car196 import Car196

    path = sys.argv[1]
    c = Car196(root=path)
    d = Dataloader(data_source=c).__getitem__(20)
    print(d)
