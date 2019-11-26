import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import functools
from plotly import express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

#fp16
# from apex import amp

import time
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('utils',''))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# alias
opj = os.path.join