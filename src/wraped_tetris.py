from typing import Optional

from tetris import Tetris
from help_learning import generate_foundation
from mino import TEMPLATE, MINO_TYPE_NUM, Mino

import torch
import numpy as np

class WrappedTetris:
    
    def __init__(self, tetris:Tetris, reset=True) -> None:
        self.tetris = tetris
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.width = self.tetris.width
        self.height = self.tetris.height
        if reset:
            self.reset()

    def get_next_mino(self) -> tuple[int, Mino]:
        return self.tetris.get_next()

    def reset(self, generate_lines = 0) -> torch.Tensor:
        self.tetris = Tetris()
        generate_foundation(self.tetris, generate_lines)
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        return self.get_current_state()

    def step(self, action:tuple[int,int]) -> tuple[float,bool]:
        '''
        input:
            action: (回転状態, x座標)
        return:
            (報酬, ゲームオーバー)
        '''
        lines_cleared, state = self.tetris.drop(action[0], action[1])
        done = state == -1
        score = (lines_cleared**2) * 5 + (self.height - self.tetris.get_max_height()) // 2
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        score /= 100
        if done:
            score = -1
        return score, done
    
    def get_current_state(self) -> torch.Tensor:
        field = self.tetris.get_field()
        return torch.FloatTensor(field).unsqueeze(0)
    
    def get_next_states(self) -> tuple[dict,dict]:
        states = {}
        rewards = {} # index: reward の辞書
        next_index, next_mino = self.get_next_mino()
        for i in range(len(next_mino.rotates)):
            rotated_mino, x_max = next_mino.rotates[i]
            for x in range(self.width - x_max):
                copied_tetris = self.tetris.deepcopy()
                copied_wrapped_tetris = WrappedTetris(copied_tetris, reset=False)
                reward, done = copied_wrapped_tetris.step((i, x))
                states[(i, x)] = copied_wrapped_tetris.get_current_state()
                rewards[(i, x)] = reward
        return states, rewards



    


def generate_wrapped_tetris():
    tetris = Tetris()
    generate_foundation(tetris, 5)
    return WrappedTetris(tetris)

if __name__ == '__main__':
    env = generate_wrapped_tetris()
    env.reset()
    print(env.get_current_state())
    for key, value in env.get_next_states().items():
        print(key)
        print(value)
