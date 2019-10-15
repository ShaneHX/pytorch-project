#!/usr/bin/env python3
# coding=UTF-8
'''
@Author: Shane
@since: 2019-08-22 10:06:48
@lastTime: 2019-10-11 16:13:53
@LastAuthor: Shane
@Description: A base class for all data loaders
'''
'''
TODO: 
[] Integration `data_prefetcher`, a method used in apex for speed up dataload
[Source code](https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256)
[Insid principle](https://github.com/NVIDIA/apex/issues/304)
'''




import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from loguru import logger
class BaseDataLoader(DataLoader):
    """[summary]
    Base class for all data loaders
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers,
                 collate_fn=default_collate):
        """[summary]

        Parameters
        ----------
        dataset : [torch.utils.data.Dataset]
            [dataset from which to load the data]
        batch_size : [int]
            [how many samples per batch to load]
        shuffle : [bool]
            [set to True to have the data reshuffled at every epoch]
        validation_split : [double or float]
            [The percentage of the validation set of the whole dataset]
        num_workers : [int]
            [how many subprocesses to use for data loading]
        collate_fn : [type], optional
            [description], by default default_collate

        Returns
        -------
        [type]
            [description]
        """
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        if(self.n_samples == 0):
            logger.error("The size of dataset is 0")
            raise IOError("Dataset error")

        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """[Generate the validation and train sampler for Dataloader
        Parameters]
        ----------
        split : [type]: double or float
            [description]: The percentage of the validation set of the dataset

        Returns
        -------
        [type]
            [description]
        """
        if(split > 0.0) and (split < 1.0):
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
