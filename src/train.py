"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample
import json

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from deep_q_network import DeepQNetwork
from wraped_tetris import generate_wrapped_tetris
from wraped_tetris import WrappedTetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="/root/src/tensorboard")
    parser.add_argument("--saved_path", type=str, default="/root/saved_models/my_model")

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    # saved_pathが存在しない場合は作成する
    if not os.path.isdir(args.saved_path):
        print("Creat foler {}".format(args.saved_path))
        os.makedirs(args.saved_path)
    else:
        print("Folder {} already exists. Will overwrite.".format(args.saved_path))
    # argsの中身をjsonファイルに書き込む
    with open(args.saved_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


def train(opt, use_learned_model=False):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = generate_wrapped_tetris()
    if use_learned_model:
        model_path = opt.saved_path+"/tetris"
        model_suffix = input("Input model suffix:")
        if model_suffix != "":
            model_path += "_{}".format(model_suffix)
        if os.path.isfile(model_path):
            print("Load model from {}".format(model_path))
            if torch.cuda.is_available():
                print("Use cuda")
                model:DeepQNetwork = torch.load(model_path)
            else:
                model:DeepQNetwork = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    try:
        foundation_num = int(input("Input the number of Ojama Blocks to stack in advance:"))
    except ValueError:
        foundation_num = 0
    state = env.reset(foundation_num)
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    last_scores = deque(maxlen=100)
    mean_scores = []
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps, next_rewards = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        for i in range(len(next_states)):
            if next_rewards[next_actions[i]] == -1:
                continue
            next_rewards[next_actions[i]] = next_rewards[next_actions[i]] * (1 - opt.gamma) + predictions[i].item() * opt.gamma
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
            next_state = next_states[index, :]
            action = next_actions[index]
        else:
            action = max(next_rewards, key=next_rewards.get)
            next_state = next_steps[action]


        reward, done = env.step(action)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset(foundation_num)
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else (1 - opt.gamma) * reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        last_scores.append((final_score, final_tetrominoes, final_cleared_lines))
        if epoch > 0 and epoch % 100 == 0:
            mean_score, mean_tetrominoes, mean_cleared_lines = np.mean(last_scores, axis=0)
            mean_scores.append(mean_score)
            print("Last 100 games mean scores: {}, mean tetrominoes: {}, mean cleared lines: {}".format(
                mean_score,
                mean_tetrominoes,
                mean_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
            plt.plot(mean_scores)
            plt.savefig("{}/mean_scores.png".format(opt.saved_path))

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    use_learned_model = "y" in input("use learned model? (y/n):")
    train(opt, use_learned_model=use_learned_model)
