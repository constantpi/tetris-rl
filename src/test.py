import argparse
import os
from random import random, randint, sample
import time
import json

import torch

from deep_q_network import DeepQNetwork
from wraped_tetris import generate_wrapped_tetris
from wraped_tetris import WrappedTetris


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--saved_path", type=str, default="/root/saved_models/my_model")
    args = parser.parse_args()

    with open(args.saved_path + '/args.json', 'r') as f:
        hyper_parameter = json.load(f)
    args.gamma = hyper_parameter["gamma"]
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    return args



def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
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
        print("Model {} does not exist.".format(model_path))
        exit()
    model.eval()
    env = generate_wrapped_tetris()
    try:
        foundation_num = int(input("Input the number of Ojama Blocks to stack in advance:"))
    except ValueError:
        foundation_num = 0
    env.reset(foundation_num)
    if torch.cuda.is_available():
        model.cuda()
    while True:
        print("\n"*20)
        time.sleep(0.1)
        next_steps, next_rewards = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        for i in range(len(next_states)):
            if next_rewards[next_actions[i]] == -1:
                continue
            next_rewards[next_actions[i]] = next_rewards[next_actions[i]] * (1 - opt.gamma) + predictions[i].item() * opt.gamma
        action = max(next_rewards, key=next_rewards.get)
        score, done = env.step(action)
        if score > 1:
            print("score:{}".format(score))
        print(env.tetris)
        if done:
            break
    print("score:{}".format(env.score))
    print("tetrominoes:{}".format(env.tetrominoes))
    print("cleared_lines:{}".format(env.cleared_lines))
        
if __name__ == "__main__":
    opt = get_args()
    test(opt)
