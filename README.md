# tetris-rl
テトリスを強化学習します。

## Requirement
Dockerがインストールされていて、とdocker-composeコマンドが使える必要があります。

## Usage
### Dockerコンテナに入る
まず、`./BUILD-DOCKER-IMAGE.sh`でDockerイメージをビルドします。

次に` ./RUN-DOCKER-CONTAINER.sh`で、Dockerコンテナを作り、そのコンテナの中に入ります。コンテナの中に入っている間は`<root>~/src$`というように表示されています。

### 学習を始める
`<root>~/src$python train.py`で学習を開始します。

初回は`use learned model? (y/n):`でnを入力してください。

そして`Input the number of Ojama Blocks to stack in advance:`では最初に積み上げておくお邪魔ブロックの段数を入力してください。0~3程度がおすすめです。

1000epochごとにモデルを/root/saved_models/my_modelに保存します。

学習を途中から再開する場合は`use learned model? (y/n):`でyを入力してから`Input model suffix:`で何epochのときのモデルから再開するか入力してください。
1000epochのときのモデルを選ぶなら1000と入力します。
また前回学習を中断させていなく更に追加学習させる場合には何も入力せずにエンターを押してください。

### モデルを試す
学習済みのモデルを実際に動かしてみたいときは`<root>~/src$python test.py`を実行してください。
1000epochのときのモデルを試したければ、`Input model suffix:`で1000と入力する必要があります。学習し終わったときのモデルを試すなら何も入力せずにエンターを押してください。

学習の際と同様に`Input the number of Ojama Blocks to stack in advance:`では最初に積み上げておくお邪魔ブロックの段数を入力してください。

## Note
`nvidia-smi`コマンドが使えるかどうかで、PCにGPUが乗っているか判断しています。このコマンドが使える場合のみ、GPUを使って学習を回します。

`<root>~/src$python train.py --help`でオプションの付け方の説明がでてきます。
lrは学習率(デフォルト0.001)、gammaは割引率(デフォルト0.8)、num_epochsは何epoch学習を回すか(デフォルト3000)です

学習の様子は/root/saved_models/my_model/mean_scores.pngに保存されます。
![image](https://github.com/constantpi/tetris-rl/assets/108005517/b2a4ea2d-5646-479e-8f0e-8e029a95c792)
![image](https://github.com/constantpi/tetris-rl/assets/108005517/9f1cfcca-53b9-4f2d-9706-0b062d29325f)

縦軸は直近の100epochにおけるゲームオーバーまでのスコアの平均で、横軸の1が100epochを表しています。

# Author
constantpi(https://github.com/constantpi/)

