# Intro Deep-RL

[![Build Status](https://travis-ci.com/henighan/deeprl-intro.svg?branch=master)](https://travis-ci.com/henighan/deeprl-intro)

## What is this?
The primary use of this repo is to store my attempts at deep reinforcement-learning algorithms, and benchmark them against the implementations in  [OpenAI's spinup](https://spinningup.openai.com/en/latest/) repository. This is my first attempt at deep reinforcement learning, hence the name.

## How can I use this?

Running this
```bash
$ deeprl -hid "(64,32)" -n 10 -env 'HalfCheetah-v2' -a ppo --epochs 750 benchmark
```
and waiting produces this

![alt text](imgs/halfcheetah_ppo_benchmark.png)

With the above line, we have run my PPO implementation against OpenAI's for 10 different random seeds. The top plot shows the history of the episode return. The bottom shows the change in the episode return over training for each implementation and each seed. To get a rough idea if the change in the return is meaningfully different between two implementations, I compare the distributions in the bottom plot with student's t-test to calculate the p-value shown.

I used [click](https://click.palletsprojects.com/en/7.x/) to make this nifty command-line-tool, which automatically gives you `--help` flags


```bash
$ deeprl --help
Usage: deeprl [OPTIONS] COMMAND [ARGS]...

  Main entry point

Options:
  -exp, --exp_name TEXT           Prefix added to experiment name
  -n, --num_runs INTEGER          Number of different random seeds to run
                                  [default: 3]
  --epochs INTEGER                Number of epochs  [default: 50]
  -steps, --steps_per_epoch INTEGER
                                  Number of epochs  [default: 4000]
  -env, --env_name TEXT           Environment name  [default: Swimmer-v2]
  -hid, --hidden_sizes TEXT       Hidden sizes for actor and critic MLPs
                                  [default: (64,64)]
  --activation TEXT               Activation to use in actor-critic MLPs
                                  [default: tanh]
  -a, --algo [vpg|trpo|ppo|ddpg|td3|sac]
                                  Algorithm (ie agent) to use  [default: vpg]
  --help                          Show this message and exit.

Commands:
  benchmark  Benchmark tom's implementation against spinup and plot
  plot       plot Logging Results
  run        Run experiment and plot Episode Return
```

If we then want to know how to use "plot", we can get help on that as well

```bash
$ deeprl plot --help
Usage: deeprl plot [OPTIONS]

  plot Logging Results

Options:
  -imp, --implementation [tom|spinup]
                                  Which implementation to run, spinup's or
                                  Tom's
  -v, --value TEXT                Value to plot  [default: AverageEpRet]
  --help                          Show this message and exit.
```

## How do I install this?

```bash
git clone git@github.com:henighan/deeprl-intro.git
cd deeprl-intro
make install
deeprl --help
```

## How do I test this?
```bash
make test
```

If you want to use the mujoco environments, you will also need to follow the instructions [here](https://spinningup.openai.com/en/latest/user/installation.html#installing-mujoco-optional).

## Other notes

Thus far I have only implemented VPG and PPO for gaussian and categorical policies, (I'm still hunting down a bug in the DDPG agent). I stole spinup's logger, so I could easily plot and compare their and my results side-by-side.
