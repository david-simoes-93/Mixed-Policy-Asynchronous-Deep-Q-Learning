# Mixed-Policy Asynchronous Deep Q-Learning
A mixed-policy version of the Asynchronous 1-step Q-learning algorithm, based on WoLF-PHC, GIGA-WoLF, WPL, EMA-QL and PGA-APP, with several Game Theoretic test scenarios.

## Multi-agent Double DQN

The Multi-agent Double DQN algorithm is in the `asyncdqn` folder. You will need [Python3.3+](https://www.python.org/download/releases/3.0/), [matplotlib](http://matplotlib.org/), [python-tk](https://wiki.python.org/moin/TkInter) and [TensorFlow 0.13+](https://www.tensorflow.org/). To run some threads locally, adjust configuration on `asyncdqn/DQN-LocalThreads.py`, and just

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
    
The algorithm works out of the box with all scenarios. This is a pseudo-code description.

![screenshot from 2017-09-06 12-22-21](https://user-images.githubusercontent.com/9117323/30109421-25f06636-92fe-11e7-9184-5a872accf2fb.png)

We test on 5 famous Game Theory challenges.

![screenshot from 2017-09-06 12-17-12](https://user-images.githubusercontent.com/9117323/30109265-a916c51a-92fd-11e7-81b4-346a48f4ec40.png)

We used a neural network with 2 hidden layers of 150 nodes each, and ELU activation. We share network weights to speed-up learning, as shown below.

![screenshot from 2017-09-06 12-17-36](https://user-images.githubusercontent.com/9117323/30109266-a9172b86-92fd-11e7-81b9-ca5c5e0facb6.png)

Below we can see the evolution of the policies of 2 agents in self-play using the Wolf-PHC,
GIGA-WoLF, WPL, and EMA-QL algorithms, over 1000 epochs of 10000 trials.
The games shown are the Tricky Game (solid) and the Biased Game (dotted),
both shown in Figure 2. Each plot represents the probability of playing the first
action by each player.

![screenshot from 2017-09-06 12-10-27](https://user-images.githubusercontent.com/9117323/30109268-a91a782c-92fd-11e7-97e1-28ab95b788d5.png)

Below we can see the evolution of the policies of 2 agents in self-play using the deep learning
implementations of Wolf-PHC, GIGA-WoLF, WPL, and EMA-QL algorithms,
over 400 epochs of 250 iterations. The games shown are the Tricky Game (solid)
and the Biased Game (dotted), both shown in Figure 2. Each plot represents
the probability of playing the first action by each player.

![screenshot from 2017-09-06 12-10-38](https://user-images.githubusercontent.com/9117323/30109267-a9177802-92fd-11e7-97c6-484033dfce20.png)
