import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import functools
from plotly import express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

opj = os.path.join