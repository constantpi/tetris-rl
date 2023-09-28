from tetris import Tetris
import random

def generate_foundation(tetris:Tetris, n:int) -> None:
    tetris.field = 0
    n = min(n, tetris.width)
    for i in range(n):
        line = (1 << tetris.width) - 1
        # line ^= 1 << (i // 2 * 2)
        # line ^= 1 << (i // 2 * 2 + 1)
        # line ^= ((1 << 2) - 1) << 4
        random_int = random.randint(0, tetris.width//2-1)
        if i % 2 == 0:
            random_int += tetris.width//2
        line ^= 1 << random_int
        # line ^= 1 << (random_int+1)
        line <<= tetris.width*i
        tetris.field |= line