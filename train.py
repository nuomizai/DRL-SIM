import numpy as np
import os
import torch
from environment import Env
from utils import seed_torch
from ppo_agent import PPOAgent
import time
from env_setting import Setting
import json
from model_test import model_test
import argparse


def main(args, env_args):
    seed_torch(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    if args.use_cuda:
        torch.cuda.set_device(args.device_num)
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    # -------------get environment information------------

    ppo_agent = []
    for i in range(args.user_num):
        ppo_agent.append(
            PPOAgent(args.state_dim, device, args.lr, args.exploration_steps,
                     args.mini_batch_num, args.use_gae, args.gamma, args.gae_param, args.ppo_epoch,
                     args.clip, args.value_coeff, args.clip_coeff, args.ent_coeff))

    done_time = 0
    episode_length = 0

    user_num = args.user_num
    env = Env(user_num, args.state_dim, device, env_args)

    action = torch.zeros(user_num, dtype=torch.long)
    value = torch.zeros(user_num)
    action_log_probs = torch.zeros(user_num)
    file_path = os.path.join(args.root_path, 'file')
    result_path = file_path + '/result.npz'
    model_path = os.path.join(args.root_path, 'model')
    os.mkdir(model_path)
    rewards = []
    server_rewards = []
    user_rewards = [[] for _ in range(user_num)]
    completion_ratio = [[] for _ in range(env.task_num)]

    ext_rewards = []
    int_rewards = []
    while True:
        if episode_length >= args.max_episode_length:
            print('training over')
            break

        print('---------------in episode ', episode_length, '-----------------------')

        step = 0
        done = True
        av_reward = torch.zeros(user_num)
        av_action = torch.zeros(user_num)

        obs = env.reset()
        for i, agent in enumerate(ppo_agent):
            agent.after_update(obs[i])

        interact_time = 0
        sum_reward = 0.0
        sum_user_reward = np.zeros(user_num)
        while step < args.exploration_steps:
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                for i, agent in enumerate(ppo_agent):
                    value[i], action[i], action_log_probs[i] = agent.act(obs[i])
                obs, reward, done = env.step(action)
            sum_reward += reward.numpy().mean()
            sum_user_reward += reward.numpy()

            av_reward += reward
            av_action += 0.2 * action.float()
            done = interact_time >= args.max_interact_time
            # ---------judge if game over --------------------
            masks = torch.tensor([[0.0] if done else [1.0]])
            # ----------add to memory ---------------------------
            for i, agent in enumerate(ppo_agent):
                agent.insert(obs[i].detach(), action[i].detach(), action_log_probs[i].detach(), value[i].detach(),
                             reward[i].detach(), masks.detach())
            step = step + 1

            if done:
                done_time += 1
                interact_time = 0
                obs = env.reset()
                for i, agent in enumerate(ppo_agent):
                    agent.reset(obs[i])

                continue

        server_reward = env.plot_server_reward(episode_length) / args.exploration_steps
        server_rewards.append(server_reward)
        ext_reward, int_reward = env.get_reward()
        ext_rewards.append(ext_reward / args.exploration_steps)
        int_rewards.append(int_reward / args.exploration_steps)
        # av_completion_ratio = env.get_completion_ratio() / args.exploration_steps

        # for i in range(env.task_num):
        #     completion_ratio[i].append(av_completion_ratio[i])
        av_value_loss = 0
        av_policy_loss = 0
        av_ent_loss = 0

        rewards.append(sum_reward / args.exploration_steps)
        for i in range(user_num):
            user_rewards[i].append(sum_user_reward[i] / args.exploration_steps)

        for i, agent in enumerate(ppo_agent):
            value_loss, policy_loss, ent_loss = agent.update(done)
            av_value_loss += value_loss
            av_policy_loss += policy_loss
            av_ent_loss += ent_loss

        av_value_loss /= user_num
        av_policy_loss /= user_num
        av_ent_loss /= user_num
        av_reward /= args.exploration_steps
        av_action /= args.exploration_steps
        episode_length += 1

    np.savez(result_path, np.asarray(rewards), np.asarray(server_rewards), np.asarray(ext_rewards),
             np.asarray(int_rewards))
    # reward_profile.close()
    for i, ppo in enumerate(ppo_agent):
        ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        torch.save(ppo.local_ppo_model.state_dict(), ppo_model_path)
    model_test(args, env_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameter setting for DRL-SIM.')
    # ------------------------------------- parameters that must be configured ---------------------------------
    parser.add_argument('--root-path', type=str, required=True, help='the path to save your results and models')
    parser.add_argument('--user-num', type=int, required=True, help='use cuda device to train models or not')

    # ------------------------------------- parameters that can be changed according to your need --------------
    parser.add_argument('--use-cuda', type=bool, default=True, help='use cuda device to train models or not')
    parser.add_argument('--device-num', type=int, default=0, help='cuda device number for training')
    parser.add_argument('--test-device-num', type=int, default=0, help='cuda device number for testing')
    parser.add_argument('--max-episode-length', type=int, default=1000)
    parser.add_argument('--max-test-length', type=int, default=100)
    parser.add_argument('--exploration-steps', type=int, default=500)
    parser.add_argument('--mini-batch-num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--max-interact-time', type=int, default=64)

    # ------------------------------------- parameters that never recommend to be changed ---------------------
    parser.add_argument('--lr', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--ent-coeff', type=float, default=0.01)
    parser.add_argument('--value-coeff', type=float, default=0.1)
    parser.add_argument('--clip-coeff', type=float, default=1.0)
    parser.add_argument('--use-gae', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_param', type=float, default=0.95)

    args = parser.parse_args()
    args.state_dim = args.user_num + 3
    local_time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
    args.root_path = os.path.join(args.root_path, local_time)
    file_path = os.path.join(args.root_path, 'file')
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, 'agent_args.txt'), 'a') as f:
        f.write(json.dumps(args.__dict__))

    env_args = Setting()
    with open(os.path.join(file_path, 'env_args.txt'), 'a') as f:
        f.write(json.dumps(env_args.__dict__))
    main(args, env_args)
