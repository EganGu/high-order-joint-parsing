from __future__ import annotations

import os
from typing import Dict, Iterable, List, Union

class Catcher(object):
    def __init__(
        self, 
        name: str
    ) -> Catcher:
        self.name = name
        self.container = []
    
    def catch(self, data):
        self.container.append(data)


class Recorder(object):
    def __init__(
        self, 
        catchers: Iterable[str], 
        path: str, 
        n_digit: int
    ) -> Recorder:
        self.catchers = catchers
        self.path = path
        self.n_digit = n_digit
    
    @classmethod
    def set_record(
        cls, 
        names: Iterable[str], 
        path: str, 
        n_digit: int = 5, 
        checkpoint: bool = False
    ) -> Recorder: 
        catchers = []
        for n in names:
            catchers.append(Catcher(n))
        if not os.path.exists(path) or not checkpoint:
            with open(path, 'w') as fw:
                fw.write(','.join(names)+'\n')
        return cls(catchers, path, n_digit)
    
    def record(self, *frame):
        frame = [round(f, self.n_digit) for f in frame]
        assert len(frame) == len(self.catchers)
        for i, c in enumerate(self.catchers):
            c.catch(frame[i])
        with open(self.path, 'a') as fw:
            fw.write(','.join([str(f) for f in frame])+'\n')
    
    def set_checkpoint(self):
        with open(self.path, 'a') as fw:
            fw.write('ck\n')
    
    @classmethod
    def filter_checkpoint(cls, from_path, to_path=None):
        with open(from_path) as fr:
            lines = fr.readlines()
        with open(to_path if to_path is not None else from_path, 'w') as fw:
            for line in lines:
                if line != 'ck\n':
                    fw.write(line)







