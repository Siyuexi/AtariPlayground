# AtariPlayground
Based on Deep Reinforcement Learning

## Requirements

Python version >= 3.7

```shell
pip install torch=1.9.0 gym=0.19.0 loguru visdom
```

## Quick Start

Run this code for "Breakout-v0" baseline training:

``` shell
python -m visdom.server
python main.py -g breakout -m train
```

Run this code for "Breakout-v0" baseline testing:

```shell
python main.py -g breakout -m test
```

If you want to use other arguments, please read "main.py" for more information.

If you want to change some hyper parameters, please read "param.yaml" for more information.
