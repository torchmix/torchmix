# Welcome to torchmix!

`torchmix` is a library that provides a collection of PyTorch modules that aims to make your code more efficient and modular. In this example, we'll show you how to use `torchmix`'s `no_parameters` context manager to speed up the creation of `Component` objects.

### Using the `no_parameters` Context Manager

If you are creating a large `Component` objects and don't need their actual parameters (e.g. weights and biases), you can use `torchmix`'s `no_parameters` context manager to speed up the creation process. This can be especially useful when you only need the `Component` objects for generating configurations and not for training or inference.

```python
import time

import torchmix
from torchmix import nn

# Create 100 Linear modules without using the no_parameters context manager
start = time.time()
for i in range(100):
    nn.Linear(10000, 10000)
print(time.time() - start)  # 42.201

# Create 100 Linear modules using the no_parameters context manager
start = time.time()
with torchmix.no_parameters():
    for i in range(100):
        nn.Linear(10000, 10000)
print(time.time() - start)  # 0.044
```

As you can see, using the `no_parameters` context manager significantly speeds up the creation process. You can use it to register as many architectural choices as you need in your `hydra` application without worrying about the performance impact.
