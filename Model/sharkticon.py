############
#Sharkticon#
############

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

BUFFER_SIZE = 1000
BATCH_SIZE = 16

logging.getLogger('tensorflow').setLevel(logging.ERROR)

