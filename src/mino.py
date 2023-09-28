'''
テトリスのミノを表すクラス
'''
TEMPLATE:list[list[tuple[int,int]]] = [
    [(0, 0), (0, 1), (1, 0), (1, 1)], #Oミノ
    [(0, 0), (0, 1), (0, 2), (0, 3)], #Iミノ
    [(0, 0), (0, 1), (0, 2), (1, 2)], #Lミノ
    [(0, 0), (1, 0), (2, 0), (2, 1)], #Jミノ
    [(0, 0), (1, 0), (1, 1), (2, 1)], #Sミノ
    [(0, 0), (0, 1), (1, 1), (1, 2)], #Zミノ
    [(0, 0), (1, 0), (1, 1), (2, 0)], #Tミノ
]
TEMPLATE = [sorted(mino) for mino in TEMPLATE]
MINO_TYPE_NUM = len(TEMPLATE)

def rotate(mino:list[tuple[int,int]]):
    ans = [(y, 3-x) for x, y in mino]
    x_min = min([x for x, y in ans])
    y_min = min([y for x, y in ans])
    ans = [(x-x_min, y-y_min) for x, y in ans]
    ans = sorted(ans)
    return ans

class Mino:
    
    def __init__(self, type:int) -> None:
        self.type = type
        self.original_mino = TEMPLATE[type]
        self.rotates = [(self.original_mino, max([x for x, y in self.original_mino]))]
        for _ in range(3):
            rotated = rotate(self.rotates[-1][0])
            x_max = max([x for x, y in rotated])
            if (rotated, x_max) in self.rotates:
                break
            self.rotates.append((rotated, x_max))

if __name__ == '__main__':
    for i in range(MINO_TYPE_NUM):
        mino = Mino(i)
        print(mino.rotates)