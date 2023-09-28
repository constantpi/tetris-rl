import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self, input_shape = (24, 10)):
        super(DeepQNetwork, self).__init__()

        self.network_list = []
        self.network_list.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
                      nn.ReLU(inplace=True)))
        self.network_list.append(nn.Flatten()) # ここで1次元に変換
        self.network_list.append(nn.Sequential(nn.Linear(in_features=16 * (input_shape[0] - 2) * (input_shape[1] - 2), out_features=256), nn.ReLU(inplace=True)))
        self.network_list.append(nn.Sequential(nn.Linear(in_features=256, out_features=1), nn.Softsign()))
        self.network_list = nn.ModuleList(self.network_list) 

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.network_list:
            x = layer(x)

        return x

if __name__ == "__main__":
    from wraped_tetris import generate_wrapped_tetris
    import torch
    env = generate_wrapped_tetris()
    deep_q_network = DeepQNetwork()
    values = []
    for key, value in env.get_next_states().items():
        print(key)
        value = value.unsqueeze(0).unsqueeze(0)
        print(value)
        values.append(value)
    values = torch.cat(values)
    print(values.shape)
    print(deep_q_network(values))
    next_actions, next_states = zip(*env.get_next_states().items())
    print(next_actions)
    print(next_states)
    next_states = torch.stack(next_states)
    next_states = next_states.unsqueeze(1)
    print(next_states.shape)