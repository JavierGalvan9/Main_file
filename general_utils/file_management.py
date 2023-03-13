#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:50:38 2021

@author: jgalvan
"""

import os
import pickle
import lzma
from pathlib import Path


def save_lzma(file, filename, parent_dir):
    """
    Function defined to save the drifting gratings 3D array into a compressed
    file in the lzma format.

    Parameters
    ----------
    file : TYPE, np.ndarray
    filename : TYPE, string
    parent_dir : TYPE, string

    Returns
    -------
    None.

    """
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    with lzma.LZMAFile(os.path.join(parent_dir, filename), 'w') as f:
        pickle.dump(file, f)
        return


def load_lzma(path):
    """
    Function defined to load the drifting gratings 3D array from a compressed
    file in the lzma format.

    Parameters
    ----------
    path : TYPE. string

    Returns
    -------
    TYPE, np.ndarray
        Three dimensional drifting gratings array

    """
    with lzma.LZMAFile(path, 'rb') as f:
        return pickle.load(f)
