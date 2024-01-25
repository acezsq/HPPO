# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import gym_hybrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
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

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=128,
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 512// 4
    # fmt: on
    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(envs.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_dis = nn.Sequential(
            layer_init(nn.Linear(envs.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 3), std=0.01),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(envs.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value_dis(self, x, action=None):
        # 接收状态x作为输入，并输出动作的未归一化的对数概率（logits）
        logits = self.actor_dis(x)
        # 使用未归一化的对数概率创建一个Categorical分布对象
        probs = Categorical(logits=logits)
        if action is None:
            # 根据概率分布随机采样一个动作
            action = probs.sample()
            # 动作（action）、动作的对数概率（log_prob(action)）、概率分布的熵（entropy()）以及通过值函数网络（critic）对状态x的值函数估计
            # 动作的对数概率（log_prob(action)）是指给定一个动作，根据策略网络输出的概率分布，计算该动作的对数概率值
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_and_value_con(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    # )
    envs = gym.make('Moving-v0')
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, envs.observation_space.shape[0])).to(device)
    actions_dis = torch.zeros((args.num_steps,)).to(device)
    logprobs_dis = torch.zeros((args.num_steps,)).to(device)

    actions_con = torch.zeros((args.num_steps, 2)).to(device)
    logprobs_con = torch.zeros((args.num_steps,2)).to(device)

    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size  # 500000 // 512 = 976

    episodic_return = 0
    episodic_length = 0
    for update in range(1, num_updates + 1):  # update 从1 到 976
        print('**************************************************')
        print('第 {} of 976 轮'.format(update))
        print('**************************************************')
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            # anneal_lr 是 Learning Rate Annealing 学习率退火
            # 学习率退火是一种训练过程中动态调整学习率的技术。它通常会在训练的早期使用较大的学习率以加快收敛速度，
            # 然后逐渐降低学习率，让模型在训练后期更加稳定地收敛或探索更细致的参数空间。
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # 执行512个step
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_dis, logprob_dis, _, value = agent.get_action_and_value_dis(next_obs)
                values[step] = value.flatten()
                action_con, logprob_con,  = agent.get_action_and_value_con(next_obs)
            actions_dis[step] = action_dis
            logprobs_dis[step] = logprob_dis
            actions_con[step] = action_con
            logprobs_con[step] = logprob_con

            # TRY NOT TO MODIFY: execute the game and log data.
            action = (action_dis.cpu(), action_con.cpu().numpy())
            next_obs, reward, done, info = envs.step(action)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(np.array(done)).to(device)
            episodic_return += reward
            episodic_length += 1
            if done == True:
                # 计算回合的奖励和长度
                next_obs = torch.Tensor(envs.reset()).to(device)
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                episodic_return = 0
                episodic_length = 0
        print('--------------------------------------------------')
        print('第 {} of 976 轮 采样完毕数据'.format(update))
        print('--------------------------------------------------')

        # bootstrap value if not done
        # 用于计算优势函数（advantages）和返回值（returns）
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            # 初始化lastgaelam变量为0，用于计算GAE（Generalized Advantage Estimation）中的累积因子
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        #
        # 相当于一共512个step的数据
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs_dis = logprobs_dis.reshape(-1)
        b_actions_dis = actions_dis.reshape(-1)
        b_logprobs_con = logprobs_con.reshape((-1, 2))
        b_actions_con = actions_con.reshape((-1, 2))

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)  # batch_size = 512   b_inds = [0,1,2....,511]
        # 用于存储每个批次的clip fraction值
        clipfracs = []
        for epoch in range(args.update_epochs):  # update_epochs = 4
            # 随机打乱b_inds数组中的元素顺序，以便每个epoch中随机选择训练样本。
            np.random.shuffle(b_inds)
            # 将训练样本划分为多个大小为args.minibatch_size = 128的小批次
            # 其中start和end是小批次的起始索引和结束索引
            # mb_inds是当前小批次中样本的索引。
            for start in range(0, args.batch_size, args.minibatch_size):  # minibatch_size = 128
                # start = 0, 128, 256, 384
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # 根据输入的观察和动作，获取新的对数概率（newlogprob），策略熵（entropy）和值函数估计值（newvalue）
                _, newlogprob_dis, entropy, newvalue = agent.get_action_and_value_dis(b_obs[mb_inds],
                                                                              b_actions_dis.long()[mb_inds])
                logratio_dis = newlogprob_dis - b_logprobs_dis[mb_inds]
                ratio_dis = logratio_dis.exp()
                # TODO
                # 重新计算一个mb_inds，这个只保留action不是2的索引
                mb_inds_con = []
                for idx in  mb_inds:
                    if action_dis[idx] != 2:
                        mb_inds_con.append(idx)
                _, newlogprob_con, = agent.get_action_and_value_con(b_obs[mb_inds_con], b_actions_con[mb_inds_con])
                newlogprob_con = newlogprob_con.gather(1, b_actions_dis[mb_inds_con]).squeeze()
                oldlogprob_con = b_logprobs_con.gather(1, b_actions_dis[mb_inds_con]).squeeze()
                logratio_con = newlogprob_con - oldlogprob_con
                ratio_con = logratio_con.exp()

                # "clip fraction"（裁剪比例）是指在使用PPO算法进行优化时，计算出的近似策略比率在被裁剪范围之外的比例。
                # 在PPO算法中，为了限制每次更新的策略变化幅度，会使用一个裁剪系数（clip coefficient）
                # 如果策略比率（新的概率与旧的概率之比）超过了裁剪系数范围之外，那么它就会被裁剪到该范围内
                # 裁剪后的策略比率被用于计算策略损失
                # "clip fraction"是指裁剪后的策略比率超过裁剪系数的比例
                # 它表示了在训练过程中有多少比例的策略比率被裁剪到了裁剪范围内
                # 通常，我们希望裁剪比例较低，即大部分策略比率都处于裁剪范围内
                # 较低的裁剪比例表明策略更新的幅度较小，收敛性更好。因此，观察和监控裁剪比例可以帮助我们了解模型训练的稳定性和效果
                # 计算旧的近似KL散度（old_approx_kl）和新的近似KL散度（approx_kl）
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio_dis).mean()
                    approx_kl = ((ratio_dis - 1) - logratio_dis).mean()
                    clipfracs += [((ratio_dis - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                mb_advantages_con = b_advantages[mb_inds_con]
                if args.norm_adv:
                    mb_advantages_con = (mb_advantages_con - mb_advantages_con.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss of dis
                pg_loss1 = -mb_advantages * ratio_dis
                pg_loss2 = -mb_advantages * torch.clamp(ratio_dis, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Policy loss of con
                pg_loss_con1 = -mb_advantages_con * ratio_con
                pg_loss_con2 = -mb_advantages_con * torch.clamp(ratio_con, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_con = torch.max(pg_loss_con1, pg_loss_con2).mean()


                # Value loss
                newvalue = newvalue.view(-1)
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

                loss_dis = pg_loss + - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss_dis.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                loss_con = pg_loss_con
                optimizer.zero_grad()
                loss_con.backward()
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

    envs.close()
    writer.close()


