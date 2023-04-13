# RLOR: A Flexible Framework of Deep Reinforcement Learning for Operation Research

:one: First work to incorporate end-to-end vehicle routing model in a modern RL platform (CleanRL)

:zap: Speed up the training of Attention Model by 8 times (25hours $\to$ 3 hours)

:mag_right: A flexible framework for developing *model*, *algorithm*, *environment*, and *search* for operation research

## News
- 13/04/2023: We release web demo on [Hugging Face 🤗](https://huggingface.co/spaces/cpwan/RLOR-TSP)!
- 24/03/2023: We release our paper on [arxiv](https://arxiv.org/abs/2303.13117)!
- 20/03/2023: We release jupyter lab demo and pretrained checkpoints!
- 10/03/2023: We release our codebase!


## Demo
We provide inference demo on colab notebook:

| Environment | Search       | Demo                                                         |
| ----------- | ------------ | ------------------------------------------------------------ |
| TSP         | Greedy       | <a target="_blank" href="https://colab.research.google.com/github/cpwan/RLOR/blob/main/demo/tsp_search.ipynb"><br/>  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/><br/></a> |
| CVRP        | Multi-Greedy | <a target="_blank" href="https://colab.research.google.com/github/cpwan/RLOR/blob/main/demo/cvrp_search.ipynb"><br/>  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/><br/></a> |




## Installation
### Conda
```shell
conda env create -n <env name> -f environment.yml
# The environment.yml was generated from
# conda env export --no-builds > environment.yml
```
It can take a few minutes.
### Optional dependency
`wandb`

Refer to their [quick start guide](https://docs.wandb.ai/quickstart) for installation.

## File structures
All the major implementations were under [rlor](./rlor) folder.
```shell
./rlor
├── envs
│   ├── tsp_data.py # load pre-generated data for evaluation
│   ├── tsp_vector_env.py # define the (vectorized) gym environment
│   ├── cvrp_data.py 
│   └── cvrp_vector_env.py 
├── models
│   ├── attention_model_wrapper.py # wrap refactored attention model to cleanRL
│   └── nets # contains refactored attention model
└── ppo_or.py # implementaion of ppo with attention model for operation research problems
```

The [ppo_or.py](./ppo_or.py) was modified from [cleanrl/ppo.py](https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/ppo.py). To see what's changed, use diff:
```shell
# apt install diff
diff --color ppo.py ppo_or.py
```

## Training OR model with PPO
### TSP
```shell
python ppo_or.py --num-steps 51 --env-id tsp-v0 --env-entry-point envs.tsp_vector_env:TSPVectorEnv --problem tsp
```
### CVRP
```shell
python ppo_or.py --num-steps 60 --env-id cvrp-v0 --env-entry-point envs.cvrp_vector_env:CVRPVectorEnv --problem cvrp
```
### Enable WandB
```shell
python ppo_or.py ... --track
```
Add `--track` argument to enable tracking with WandB.

### Where is the tsp data?
It can be generated from the [official repo](https://github.com/wouterkool/attention-learn-to-route) of the attention-learn-to-route paper. You may modify the [./envs/tsp_data.py](./envs/tsp_data.py) to update the path to data accordingly.

# Acknowledgements
The neural network model is refactored and developed from [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route).

The idea of multiple trajectory training/ inference is from [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html).

The RL environments are defined with [OpenAI Gym](https://github.com/openai/gym/tree/0.23.1).

The PPO algorithm implementation is based on [CleanRL](https://github.com/vwxyzjn/cleanrl).
