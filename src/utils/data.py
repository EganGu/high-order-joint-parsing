# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterable, Union
import pathos.multiprocessing as mp
import torch.distributed as dist
from supar.utils.fn import binarize, debinarize, kmeans
from supar.utils.logging import logger, progress_bar
from supar.utils.parallel import is_master
from supar.utils.transform import Transform
from supar.utils.data import Dataset, DataLoader, Sampler
from .transform import JointBatch
from supar.utils.common import INF


class JointDataset(Dataset):

    def __init__(
        self,
        transform: Transform,
        data_dep: Union[str, Iterable],
        data_con: Union[str, Iterable],
        cache: bool = False,
        binarize: bool = False,
        max_len: int = None,
        bin: str = None,
        encoder: str = '',
        bert: str = '',
        **kwargs
    ) -> Dataset:
        super(Dataset, self).__init__()

        self.transform = transform
        self.data_dep = data_dep
        self.data_con = data_con
        self.cache = cache
        self.binarize = binarize
        self.kwargs = kwargs
        self.max_len = max_len or INF
        self.bin = bin

        if cache:
            if not isinstance(data_dep, str) or not os.path.exists(data_dep) or \
               not isinstance(data_con, str) or not os.path.exists(data_con):
                raise FileNotFoundError(
                    "Only files are allowed for binarization, but not found")

            # if self.bin is None:
            # fbin_suffix = ('.'+bert) if encoder == 'bert' else ''
            if 'tag' in self.kwargs['feat']:
                self.fbin = data_dep + \
                    f".{self.kwargs['encoder_type']}.{self.kwargs['binarize_way']}.POS.joint.new.pt"
            else:
                self.fbin = data_dep + \
                    f".{self.kwargs['encoder_type']}.{self.kwargs['binarize_way']}.joint.new.pt"

            # else:
            #     os.makedirs(self.bin, exist_ok=True)
            #     self.fbin = os.path.join(self.bin, os.path.split(data_dep)[1]) + '.joint.pt'

            if self.binarize or not os.path.exists(self.fbin):
                logger.info(f"Seeking to cache the data to {self.fbin} first")
            else:
                try:
                    self.sentences = debinarize(
                        self.fbin, meta=True)['sentences']
                except Exception:
                    raise RuntimeError(f"Error found while debinarizing {self.fbin}, which may have been corrupted. "
                                       "Try re-binarizing it first")
        else:
            self.sentences = list(transform.load(data_dep, data_con, **kwargs))

    def __getattr__(self, name):
        if name not in {f.name for f in self.transform.flattened_fields}:
            raise AttributeError(f'{name} not in transform.flattened_fields. ')
        if self.cache:
            if os.path.exists(self.fbin) and not self.binarize:
                sentences = self
            else:
                sentences = self.transform.load(
                    self.data_dep, self.data_con, **self.kwargs)
            return (getattr(sentence, name) for sentence in sentences)
        return [getattr(sentence, name) for sentence in self.sentences]

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True,
        chunk_size: int = 1000,
    ) -> Dataset:
        # numericalize all fields
        if not self.cache:
            self.sentences = [i for i in self.transform(
                self.sentences) if len(i) < self.max_len]
        else:
            # if not forced to do binarization and the binarized file already exists, directly load the meta file
            if os.path.exists(self.fbin) and not self.binarize:
                self.sentences = debinarize(self.fbin, meta=True)['sentences']
            else:
                @contextmanager
                def cache(sentences):
                    ftemp = tempfile.mkdtemp()
                    fs = os.path.join(ftemp, 'sentences')
                    fb = os.path.join(ftemp, os.path.basename(self.fbin))
                    global global_transform
                    global_transform = self.transform
                    sentences = binarize({'sentences': progress_bar(sentences)}, fs)[
                        1]['sentences']
                    try:
                        yield ((sentences[s:s+chunk_size], fs, f"{fb}.{i}", self.max_len)
                               for i, s in enumerate(range(0, len(sentences), chunk_size)))
                    finally:
                        del global_transform
                        shutil.rmtree(ftemp)

                def numericalize(sentences, fs, fb, max_len):
                    sentences = global_transform(
                        (debinarize(fs, sentence) for sentence in sentences))
                    sentences = [i for i in sentences if len(i) < max_len]
                    return binarize({'sentences': sentences, 'sizes': [sentence.size for sentence in sentences]}, fb)[0]

                logger.info(f"Seeking to cache the data to {self.fbin} first")
                # numericalize the fields of each sentence
                if is_master():
                    with cache(self.transform.load(self.data_dep, self.data_con, **self.kwargs)) as chunks, mp.Pool(32) as pool:
                        results = [pool.apply_async(
                            numericalize, chunk) for chunk in chunks]
                        self.sentences = binarize((r.get() for r in results), self.fbin, merge=True)[
                            1]['sentences']
                if dist.is_initialized():
                    dist.barrier()
                if not is_master():
                    self.sentences = debinarize(
                        self.fbin, meta=True)['sentences']
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans(self.sizes, n_buckets)))
        self.loader = DataLoader(transform=self.transform,
                                 dataset=self,
                                 batch_sampler=Sampler(
                                     self.buckets, batch_size, shuffle, distributed),
                                 num_workers=n_workers,
                                 collate_fn=collate_fn_joint,
                                 pin_memory=pin_memory)
        return self


def collate_fn_joint(x):
    return JointBatch(x)
