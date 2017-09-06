# Mixed-Policy Asynchronous Deep Q-Learning
A multi-agent version of the Asynchronous 1-step Q-learning algorithm, with several Game Theoretic test scenarios.

## Multi-agent Double DQN

The Multi-agent Double DQN algorithm is in the `MaDDQN` folder. You will need [Python3.3+](https://www.python.org/download/releases/3.0/), [matplotlib](http://matplotlib.org/), [python-tk](https://wiki.python.org/moin/TkInter) and [TensorFlow 0.13+](https://www.tensorflow.org/). To run some threads locally, adjust configuration on `asyncdqn/DQN-LocalThreads.py`, and just

    export PYTHONPATH=$(pwd)
    python3 asyncdqn/DQN-LocalThreads.py
    
To run 15 processes distributed, adjust configuration on `asyncdqn/DQN-Distributed.py`, and

    ./start-dqn-mixed.sh 15
    
If you want to test table-based mixed-policy algorithms, you can also adjust configuration on `mixedQ/run_mixed_algorithms.py`, and

    export PYTHONPATH=$(pwd)
    python3 mixedQ/run_mixed_algorithms.py
    
For specific tests of WPL in a multi-state environment, configure `mixedQ/wpl_nrps.py` and

    export PYTHONPATH=$(pwd)
    python3 mixedQ/wpl_nrps.py
    
## Results
    
The algorithm works out of the box with all scenarios. We used a neural network with 2 hidden layers of 150 nodes each, and ELU activation.
