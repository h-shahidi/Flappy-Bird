# Flappy-Bird

* [Original Repo](https://github.com/yanpanlau/Keras-FlappyBird)

# Dependencies

* Keras
* pygame
* scikit-image
* h5py

# Training

```
python main.py -m "train" --training_algorithm "DQN"
```


# Running

```
python main.py -m "run" --training_algorithm "DQN"
```

# Available Training Algorithms
`main.py` supports DQN, double DQN, DQN with upper confidence bound (UCB), bootstrapped DQN, and bootstrapped DQN with UCB. For [asynchronous one-step Q-learning](https://arxiv.org/pdf/1602.01783.pdf), please run:

```
python Async.py
```

