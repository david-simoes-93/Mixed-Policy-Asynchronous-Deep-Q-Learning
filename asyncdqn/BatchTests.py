import time
import os
import signal
import subprocess

python_str = '/home/david/.virtualenvs/tf/bin/python3 DQN-LocalThreads.py'

def call_pursuit(cmd, pursuit_mod=''):
    os.system(cmd)

if True:
    for threads in range(1, 24):
        for predators in range(2, 24):
            os.system(python_str + ' --num_slaves='+str(threads)+' --num_predators='+str(predators))
