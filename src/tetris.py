from typing import Optional, Deque, Self
import numpy as np
import random
import copy
import itertools
from collections import deque
import mino

class Tetris:

    def __init__(self,
                 MINO_TYPE_NUM = mino.MINO_TYPE_NUM,
                 height = 20,
                 width = 10,
                 field = 0,
                 out_of_field:Optional[int] = None,
                 bottom:Optional[int] = None,
                 mino_template:list[mino.Mino] = [mino.Mino(i) for i in range(mino.MINO_TYPE_NUM)],
                 nexts:Optional[deque[int]]=None) -> None:
        
        self.MINO_TYPE_NUM = MINO_TYPE_NUM #ミノの種類数
        self.height = height #画面の高さ
        self.width = width #画面の幅
        self.field = field #画面の状態を表すビット列。埋まっている場所は1、空いている場所は0
        if out_of_field is None:
            self.out_of_field = ((1 << self.width*4) - 1) << self.width*self.height #画面上部の4行
        else:
            self.out_of_field = out_of_field
        if bottom is None:
            self.bottom = self.generate_line(0) #画面の一番下のライン
        else:
            self.bottom = bottom
        self.mino_template = mino_template #ミノのテンプレート
        if nexts is None:
            self.nexts:deque[int] = deque() #次に落ちてくるミノのリスト
            self.generate_nexts()
            self.generate_nexts()
        else:
            self.nexts = nexts
        self.index_field = np.array(range(self.width*(self.height+4) -1, -1, -1)).reshape(self.height+4, self.width)

    def drop(self, mino_rotate:int, mino_x:int, do_erase=True) -> tuple[int,int]:
        '''
        ミノを落とす
        input:
            mino_rotate: ミノの回転状態
            mino_x: ミノのx座標
        return:
            (消したライン数, 状態)
            状態は0が通常、-1がゲームオーバー、-2が異常
        '''
        mino_rotate = int(mino_rotate)
        mino_x = int(mino_x)
        mino_type = self.nexts.popleft()
        state = 0
        erased_lines = 0
        if len(self.nexts) <= self.MINO_TYPE_NUM:
            self.generate_nexts()
        try:
            mino, x_max = self.mino_template[mino_type].rotates[mino_rotate]
        except IndexError:
            state = -2
            mino, x_max = random.choice(self.mino_template[mino_type].rotates)
        if x_max + mino_x >= self.width:
            state = -2
            mino_x = random.randint(0, self.width - x_max - 1)
        mino = [(self.width - (1 + x + mino_x), self.height + 3 - y) for x, y in mino]
        mino_bit = 0
        for x, y in mino:
            mino_bit |= 1 << (self.width*y + x)
        block = (self.field << self.width) | self.bottom
        for i in range(self.height+6):
            if block & mino_bit != 0:
                break
            mino_bit >>= self.width
        self.field |= mino_bit
        if do_erase:
            erased_num = self.erase()
            if self.is_gameover():
                state = -1
        else:
            erased_num = 0
        return erased_num, state
    
    def erase(self) -> int: #消したライン数を返す
        erased = 0
        i = 0
        while i < self.height+4:
            line = self.generate_line(i)
            overlap = self.field & line
            if overlap == 0:
                break
            if overlap == line:
                upper = self.field >> self.width*(i+1)
                lower = self.field & ((1 << self.width*i) - 1)
                self.field = (upper << self.width*i) | lower
                erased += 1
            else:
                i += 1
        return erased

    def generate_nexts(self) -> None:
        nexts = random.sample(range(self.MINO_TYPE_NUM), self.MINO_TYPE_NUM)
        self.nexts.extend(nexts)

    def generate_line(self, line:int) -> int:
        return ((1 << self.width) - 1) << self.width*line

    def is_gameover(self) -> bool:
        return self.field & self.out_of_field != 0
    
    def get_next(self) -> tuple[int, mino.Mino]:
        next_index = self.nexts[0]
        return next_index, self.mino_template[next_index]
    
    def get_nexts_index(self) -> list[int]:
        return np.array(self.nexts)[:self.MINO_TYPE_NUM]
    
    def seed(self, seed:Optional[int]) -> None:
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

    def get_field(self) -> np.ndarray:
        state = np.zeros((self.height + 4, self.width), dtype=np.int8)
        for (i, j) in itertools.product(range(self.height + 4), range(self.width)):
            state[i, j] = (self.field >> int(self.index_field[i, j])) & 1
        return state

    def deepcopy(self) -> Self:
        nexts = copy.deepcopy(list(self.nexts))
        tetris = Tetris(MINO_TYPE_NUM=self.MINO_TYPE_NUM,
                        height=self.height,
                        width=self.width,
                        field=self.field,
                        out_of_field=self.out_of_field,
                        bottom=self.bottom,
                        mino_template=self.mino_template,
                        nexts=deque(nexts))
        return tetris
    
    def get_max_height(self) -> int:
        for i in range(self.height+4):
            line = self.generate_line(i)
            overlap = self.field & line
            if overlap == 0:
                return i
        return self.height+4

    def __str__(self) -> str:
        ans = ''
        for i in range(self.height-1, -1, -1):
            line = self.generate_line(i)
            overlap = self.field & line
            overlap >>= self.width*i
            ans += str(bin(overlap))[2:].zfill(self.width) + '\n'
        ans = ans.replace('0', '.')
        ans = ans.replace('1', 'O')
        return ans
    


if __name__ == '__main__':
    tetris = Tetris()
    for i in range(100):
        print(i)
        print(tetris)
        next_index, next_mino = tetris.get_next()
        rotate = random.randint(0, len(next_mino.rotates)-1)
        rotated_mino, x_max = next_mino.rotates[rotate]
        x = random.randint(0, tetris.width - x_max - 1)
        erased_lines, state = tetris.drop(4, 100)
        if state == -1:
            print("gameover")
            break
        if erased_lines > 0:
            print("erased", erased_lines, "lines")
    print(tetris)
    print(tetris.get_nexts_index())
    print(tetris.get_field())