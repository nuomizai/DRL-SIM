import numpy as np
import os
import torch
from environment import Env
from utils import seed_torch
from ppo_agent import PPOAgent
import csv


def model_test(args, env_args):
    seed_torch(args.seed)

    print('in test process')
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.test_device_num == -1:
        test_device_name = 'cpu'

    else:
        test_device_name = 'cuda:' + str(args.test_device_num)
        torch.cuda.set_device(args.test_device_num)

    # -------------get environment information------------
    ppo_agent = []
    for i in range(args.user_num):
        ppo_agent.append(
            PPOAgent(args.state_dim, test_device_name, args.lr, args.exploration_steps,
                     args.mini_batch_num, args.use_gae, args.gamma, args.gae_param, args.ppo_epoch,
                     args.clip, args.value_coeff, args.clip_coeff, args.ent_coeff))

    ori_device_name = 'cuda:' + str(args.device_num)
    model_path = os.path.join(args.root_path, 'model')
    for i, agent in enumerate(ppo_agent):
        ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        agent.load_model(ppo_model_path, ori_device_name, test_device_name)
        agent.local_ppo_model.eval()

    done_time = 0
    episode_length = 0

    user_num = args.user_num
    env = Env(user_num, args.state_dim, test_device_name, env_args)

    action = torch.zeros(user_num, dtype=torch.long)
    final_av_reward = 0
    final_av_server_reward = 0
    test_file_path = os.path.join(args.root_path, 'test_file')
    if not os.path.exists(test_file_path):
        os.mkdir(test_file_path)
    test_result_profile = open(test_file_path + '/test_result.csv', 'w', newline='')
    test_writer = csv.writer(test_result_profile)

    av_ext_reward = 0
    av_int_rewards = 0

    av_completion_ratio = 0

    result_path = test_file_path + '/test_result.npz'
    # -----------------------------------------

    all_remaining_energy = []
    while True:
        if episode_length >= args.max_test_length:
            print('training over')
            break

        print('---------------in episode ', episode_length, '-----------------------')

        step = 0
        done = True
        av_reward = 0
        av_action = torch.zeros(user_num)

        obs = env.reset()
        interact_time = 0
        remaining_energy = []
        remaining_energy.append(env.remain_energy.mean())
        while step < args.exploration_steps:
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                for i, agent in enumerate(ppo_agent):
                    _, action[i], _ = agent.act(obs[i])
                obs, reward, done = env.step(action)
            remaining_energy.append(env.remain_energy.mean())
            av_reward += np.mean(reward.numpy())
            av_action += 0.2 * action.float()

            step = step + 1
            done = interact_time >= args.max_interact_time
            if done:
                # env.draw_remain_energy(done_time)
                done_time += 1
                interact_time = 0
                obs = env.reset()
                if len(remaining_energy) == args.max_interact_time + 1 and len(all_remaining_energy) < 100:
                    all_remaining_energy.append(remaining_energy)
                remaining_energy = []
                remaining_energy.append(env.remain_energy.mean())
                for i, agent in enumerate(ppo_agent):
                    agent.reset(obs[i])

                continue
        av_reward /= args.exploration_steps
        ext_reward, int_reward = env.get_reward()

        av_ext_reward += ext_reward / args.exploration_steps
        av_int_rewards += int_reward / args.exploration_steps

        completion_ratio = env.get_completion_ratio()
        av_completion_ratio += completion_ratio

        final_av_reward += av_reward
        final_av_server_reward += env.plot_server_reward(episode_length) / args.exploration_steps
        episode_length += 1

    test_writer.writerow(
        ['vehicle reward', 'server reward', 'extrinsic reward', 'intrinsic reward', 'completion ratio'])
    test_writer.writerow([final_av_reward / args.max_test_length, final_av_server_reward / args.max_test_length,
                          av_ext_reward / args.max_test_length, av_int_rewards / args.max_test_length,
                          av_completion_ratio / args.max_test_length])
    test_result_profile.close()
    np.savez(result_path, np.asarray(all_remaining_energy))
    print('Finish! Results saved in ', args.root_path)

