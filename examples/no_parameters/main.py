import time

import torchmix
from torchmix import nn

start = time.time()
for i in range(100):
    nn.Linear(10000, 10000)
print(time.time() - start)  # 42.201

start = time.time()
with torchmix.no_parameters():
    for i in range(100):
        nn.Linear(10000, 10000)
print(time.time() - start)  # 0.044
