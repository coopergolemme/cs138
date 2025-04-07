# Comparing Uniform, On-Policy, and MCTS Approaches in Tabular Reinforcement Learning

This repo contains the files used to generate the figures in my report comparing uniform, on-policy and MCTS approaches to tabular RL.

## MCTS.py

Contains code creating a tree-like MDP with a given depth and branching factor as well as methods to perform Monte Carlo Tree Search on this MDP. Also contains a few functions to test the affects of varying different parameters. You can run this file by running.

```bash
python MCTS.py
```

This should create a plot similar to the ones in the report. Try varying various parameters in the main function of the program to create MDPs with different parameters and to vary the MCTS algorithm.

## MDP.py

Contains code for creating MDP class, that is a stochastic Markov Decision Process with a specified number of states and a specified branching factor.

This file also contains methods to run experiments and perform value iteration. Experiments compare on-policy and uniform update patterns like described in the report.

You can run this file as

```bash
python MDP.py
```

to create the first few figures in the report. Try varying the experiment parameters at the top of the main function to create MDPs with different branching factors.

# Requirements

```bash
pip install numpy matplotlib tqdm concurrent math random
```
