import random
import numpy as np

prompt_algos = [
    "io", 
    "sc", 
    "cot", 
    "tot", 
    "minimax", 
    "abyssal", 
    'max_power',
    'one_step',
    'random'
    ]

PNUMBER1 = str(np.random.randint(0,10000))
print(PNUMBER1)
seed = 100
random.seed(seed)
np.random.seed(seed)