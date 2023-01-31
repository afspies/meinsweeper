# ----------------------------------------------------------------------------------------------------------------------
# Asychronous timeout iterator
# Taken from  https://stackoverflow.com/questions/50241696/how-to-iterate-over-an-asynchronous-iterator-with-a-timeout
from typing import *
import asyncio
T = TypeVar('T')
# async generator, needs python 3.6
async def timeout_iterator(it: AsyncIterator[T], timeo: float, sentinel: T) -> AsyncGenerator[T, None]:
    try:
        nxt = asyncio.ensure_future(it.__anext__())
        while True:
            try:
                yield await asyncio.wait_for(asyncio.shield(nxt), timeo)
                nxt = asyncio.ensure_future(it.__anext__())
            except asyncio.TimeoutError:
                yield sentinel
    except StopAsyncIteration:
        pass
    finally:
        nxt.cancel()  # in case we're getting cancelled our self
# ----------------------------------------------------------------------------------------------------------------------
import itertools
# Taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
# ----------------------------------------------------------------------------------------------------------------------
# Random Stuff
import os
from pathlib import Path
from contextlib import contextmanager
# Taken from https://stackoverflow.com/questions/41742317/how-can-i-change-directory-with-python-pathlib
@contextmanager
def set_directory(path: Path):
    # Sets the cwd within the contex
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

from datetime import datetime
from time import time
def get_time_diff(start_time):
    # It took me almost 30 minutes to get this working
    # I'm a bloody idiot 
    # Expects starttime to be a datetime time
    if isinstance(start_time, float):
        start_time = datetime.fromtimestamp(start_time)
    dt = datetime.fromtimestamp(time()) - start_time
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    f = ''
    if hours > 0:
        f += f'{hours} Hours, '
    if minutes > 0:
        f += f'{minutes} Minutes '
    if  (hours + minutes) > 0:
        f += 'and '
    f += f'{seconds} Seconds'
    return f 

# Taken from https://github.com/XuehaiPan/nvitop/blob/main/nvitop/core/utils.py
KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30
TiB = 1 << 40
PiB = 1 << 50
NA = -1
def bytes2human(x):  # pylint: disable=too-many-return-statements
    if x is None or x == NA:
        return NA

    if not isinstance(x, int):
        try:
            x = round(float(x))
        except ValueError:
            return NA

    if x < KiB:
        return f'{x}B'
    if x < MiB:
        return f'{round(x / KiB)}KiB'
    if x <= 20 * GiB:
        return f'{round(x / MiB)}MiB'
    if x < 100 * GiB:
        return '{:.2f}GiB'.format(round(x / GiB, 2))
    if x < 1000 * GiB:
        return '{:.1f}GiB'.format(round(x / GiB, 1))
    if x < 100 * TiB:
        return '{:.2f}TiB'.format(round(x / TiB, 2))
    if x < 1000 * TiB:
        return '{:.1f}TiB'.format(round(x / TiB, 1))
    if x < 100 * PiB:
        return '{:.2f}PiB'.format(round(x / PiB, 2))
    return '{:.1f}PiB'.format(round(x / PiB, 1))
# ----------------------------------------------------------------------------------------------------------------------