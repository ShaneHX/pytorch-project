#!/usr/bin/env python3
# coding=UTF-8
'''
@Brief: 
@Author: Shane
@Version: 
@since: 2019-08-23 13:22:18
@lastTime: 2019-08-23 13:31:00
@LastAuthor: Shane
'''
import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

'''
TODO: 
+ Add pytorch summary: https://github.com/sksq96/pytorch-summary
'''
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
