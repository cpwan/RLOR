# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import shutil
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--problem", type=str, default="cvrp",
        help="the OR problem we are trying to solve, it will be passed to the agent")
    parser.add_argument("--env-id", type=str, default="cvrp-v0",
        help="the id of the environment")
    parser.add_argument("--env-entry-point", type=str, default="envs.cvrp_vector_env:CVRPVectorEnv",
        help="the path to the definition of the environment, for example `envs.cvrp_vector_env:CVRPVectorEnv` if the `CVRPVectorEnv` class is defined in ./envs/cvrp_vector_env.py")
    parser.add_argument("--total-timesteps", type=int, default=6_000_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1024,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--n-traj", type=int, default=50,
        help="number of trajectories in a vectorized sub-environment")
    parser.add_argument("--n-test", type=int, default=1000,
        help="how many test instance")
    parser.add_argument("--multi-greedy-inference", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to use multiple trajectory greedy inference")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


from wrappers.recordWrapper import RecordEpisodeStatistics


def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    os.makedirs(os.path.join(f"runs/{run_name}", "ckpt"), exist_ok=True)
    shutil.copy(__file__, os.path.join(f"runs/{run_name}", "main.py"))
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #######################
    #### Env defintion ####
    #######################

    gym.envs.register(
        id=args.env_id,
        entry_point=args.env_entry_point,
    )

    # training env setup
    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    # evaluation env setup: 1.) from a fix dataset, or 2.) generated with seed
   # 1.) use test instance from a fix dataset
    test_envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                cfg={"eval_data": True, "eval_partition": "eval", "eval_data_idx": i},
            )
            for i in range(args.n_test)
        ]
    )
#     # 2.) use generated evaluation instance instead
#     import logging
#     logging.warning('Using generated evaluation instance. For benchmarking, please download the fix dataset.')
#     test_envs = SyncVectorEnv([make_env(args.env_id, args.seed + args.num_envs + i) for i in range(args.n_test)])
    
    assert isinstance(
        envs.single_action_space, gym.spaces.MultiDiscrete
    ), "only discrete action space is supported"

    #######################
    ### Agent defintion ###
    #######################

    agent = Agent(device=device, name=args.problem).to(device)
    # agent.backbone.load_state_dict(torch.load('./vrp50.pt'))
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay
    )

    #######################
    # Algorithm defintion #
    #######################

    # ALGO Logic: Storage setup
    obs = [None] * args.num_steps
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates + 1):
        agent.train()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        next_obs = envs.reset()
        encoder_state = agent.backbone.encode(next_obs)
        next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
        r = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value_cached(
                    next_obs, state=encoder_state
                )
                action = action.view(args.num_envs, args.n_traj)
                values[step] = value.view(args.num_envs, args.n_traj)
            actions[step] = action
            logprobs[step] = logprob.view(args.num_envs, args.n_traj)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = next_obs, torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    r.append(item)
        print("completed_episodes=", len(r))
        avg_episodic_return = np.mean([rollout["episode"]["r"].mean() for rollout in r])
        max_episodic_return = np.mean([rollout["episode"]["r"].max() for rollout in r])
        avg_episodic_length = np.mean([rollout["episode"]["l"].mean() for rollout in r])
        print(
            f"[Train] global_step={global_step}\n \
            avg_episodic_return={avg_episodic_return}\n \
            max_episodic_return={max_episodic_return}\n \
            avg_episodic_length={avg_episodic_length}"
        )
        writer.add_scalar("charts/episodic_return_mean", avg_episodic_return, global_step)
        writer.add_scalar("charts/episodic_return_max", max_episodic_return, global_step)
        writer.add_scalar("charts/episodic_length", avg_episodic_length, global_step)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value_cached(next_obs, encoder_state).squeeze(-1)  # B x T
            advantages = torch.zeros_like(rewards).to(device)  # steps x B x T
            lastgaelam = torch.zeros(args.num_envs, args.n_traj).to(device)  # B x T
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done  # next_done: B
                    nextvalues = next_value  # B x T
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]  # B x T
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = {
            k: np.concatenate([obs_[k] for obs_ in obs]) for k in envs.single_observation_space
        }

        # Edited
        b_logprobs = logprobs.reshape(-1, args.n_traj)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1, args.n_traj)
        b_returns = returns.reshape(-1, args.n_traj)
        b_values = values.reshape(-1, args.n_traj)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]  # mini batch env id
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                r_inds = np.tile(np.arange(envsperbatch), args.num_steps)

                cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
                encoder_state = agent.backbone.encode(cur_obs)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                    {k: v[mb_inds] for k, v in b_obs.items()},
                    b_actions.long()[mb_inds],
                    (embedding[r_inds, :] for embedding in encoder_state),
                )
                # _, newlogprob, entropy, newvalue = agent.get_action_and_value({k:v[mb_inds] for k,v in b_obs.items()}, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # Value loss
                newvalue = newvalue.view(-1, args.n_traj)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update % 1000 == 0 or update == num_updates:
            torch.save(agent.state_dict(), f"runs/{run_name}/ckpt/{update}.pt")
        if update % 100 == 0 or update == num_updates:
            agent.eval()
            test_obs = test_envs.reset()
            r = []
            for step in range(0, args.num_steps):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logits = agent(test_obs)
                if step == 0:
                    if args.multi_greedy_inference:
                        if args.problem == 'tsp':
                            action = torch.arange(args.n_traj).repeat(args.n_test, 1)
                        elif args.problem == 'cvrp':
                            action = torch.arange(1, args.n_traj + 1).repeat(args.n_test, 1)
                # TRY NOT TO MODIFY: execute the game and log data.
                test_obs, _, _, test_info = test_envs.step(action.cpu().numpy())

                for item in test_info:
                    if "episode" in item.keys():
                        r.append(item)

            avg_episodic_return = np.mean([rollout["episode"]["r"].mean() for rollout in r])
            max_episodic_return = np.mean([rollout["episode"]["r"].max() for rollout in r])
            avg_episodic_length = np.mean([rollout["episode"]["l"].mean() for rollout in r])
            print(f"[test] episodic_return={max_episodic_return}")
            writer.add_scalar("test/episodic_return_mean", avg_episodic_return, global_step)
            writer.add_scalar("test/episodic_return_max", max_episodic_return, global_step)
            writer.add_scalar("test/episodic_length", avg_episodic_length, global_step)

    envs.close()
    writer.close()
