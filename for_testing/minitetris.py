import torch
from torch import nn
from torch import optim
from collections import namedtuple, deque
import numpy as np
import random
import time
from torch.nn.modules.batchnorm import BatchNorm2d

from torch.nn.modules.pooling import MaxPool2d

# Tetris game functions

game_rows = 10
game_cols = 7

def piece_step(game, pos, lower_pos):
    if not(piece_landed(game, lower_pos)):
        for i in pos:
            game[i[0], i[1]] = 0
            i[0] += 1
        for i in pos:
            game[i[0], i[1]] = 1
        for i in lower_pos:
            i[0] += 1


def create_piece(game):
    pos = [[0, 4]]
    lower_pos = [[0, 4]]
    return pos, lower_pos


def piece_landed(game, lower_pos):
    for i in lower_pos:
        if i[0]+1 == game.shape[0] or game[i[0]+1, i[1]] == 1:
            return True


# if a piece lands in row 3 its game over
def game_over(lower_pos):
    return True if lower_pos[0][0] <= 1 else False


def score(game):
    return np.sum(game[-1])

    # reward = 0
    # for row in range(game.shape[0]):
    #     scored_row = True
    #     for col in range(game.shape[1]):
    #         if(game[row, col] == 0):
    #             scored_row = False
    #             break
    #     if(scored_row):
    #         reward += 1
    #         game[row, :] = 0
    #         for i in reversed(range(0, row)):
    #             for j in range(game.shape[1]):
    #                 if(game[i, j] == 1):
    #                     game[i, j] = 0
    #                     game[i+1, j] = 1
    # return reward


def move_left(game, pos, lower_pos):
    possible = True
    for i in pos:
        if i[1] == 0:
            possible = False
    if possible:
        if game[lower_pos[-1][0], lower_pos[-1][1]-1] == 1:
            possible = False
    if possible:
        for i in pos:
            game[i[0], i[1]] = 0
        for i in pos:
            i[1] -= 1
            game[i[0], i[1]] = 1
        for i in lower_pos:
            i[1] -= 1


def move_right(game, pos, lower_pos):
    possible = True
    for i in pos:
        if i[1] == game.shape[1]-1:
            possible = False
    if possible:
        if game[lower_pos[-1][0], lower_pos[-1][1]+1] == 1:
            possible = False
    if possible:
        for i in pos:
            game[i[0], i[1]] = 0
        for i in pos:
            i[1] += 1
            game[i[0], i[1]] = 1
        for i in lower_pos:
            i[1] += 1


def move_piece(game, pos, lower_pos, action):
    if action == 1:
        move_left(game, pos, lower_pos)
    elif action == 2:
        move_right(game, pos, lower_pos)
    piece_step(game, pos, lower_pos)


# Neural Network classes and functions
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


def numpy_to_tensor(nparray):
    tmp = np.copy(nparray)
    tensor = torch.from_numpy(tmp)
    tensor = tensor.to(dtype=torch.float)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = torch.unsqueeze(tensor, 0)
    return tensor


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DqnNet(nn.Module):
    def __init__(self):
        super(DqnNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Tanh()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(16 * game_rows * game_cols, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 3),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x)
        x = self.linear_layers(x)
        return x


def select_action(state):
    global epsilon
    rand = random.random()
    if rand <= epsilon:
        action = random.randrange(0, 3, 1)
        if epsilon > epsilon_end: epsilon *= decay
    else:
        with torch.no_grad():
            predictions = policy_net(state)
        action = torch.argmax(predictions).item()
    return action


def update(loss_function, optimizer):
    if(len(memory) < 64):
        return
    target = []
    prediction = []
    replays = memory.sample(64)
    for replay in replays:
        prediction.append(torch.unsqueeze(
            policy_net(replay.state)[replay.action], 0))
        if replay.reward < 0:
            target.append(torch.tensor([replay.reward]))
        else:
            target.append(torch.unsqueeze(replay.reward + discount *
                                          max(target_net(replay.next_state)), 0))
    # print(target)
    # print(prediction)
    target = torch.cat([t for t in target])
    prediction = torch.cat([t for t in prediction])
    # print(f"target: {target}")
    # print(f"prediction: {prediction}")
    loss = loss_function(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# initializations for the neural network
random.seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy_net = DqnNet()
policy_net.to(device)
target_net = DqnNet()
target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
loss_function = nn.SmoothL1Loss()  # Huber Loss
optimizer = optim.RMSprop(policy_net.parameters())
memory = ExperienceReplay(100000)

epsilon = 1
epsilon_end = 0.05
decay = 0.999
discount = 0.9

game = np.zeros((game_rows, game_cols))
pos, lower_pos = create_piece(game)

update_iter = 0
update_step = 64
copy_weights_iter = 0
copy_weights_step = 512

lost = 0
scored = 0
iteration = 0
loss = 0

while(True):
    reward = 0
    old_state = np.copy(game)
    action = select_action(numpy_to_tensor(game))
    move_piece(game, pos, lower_pos, action)
    if(piece_landed(game, lower_pos)):
        if(game_over(lower_pos)):
            reward = -10
            game = np.zeros((game_rows, game_cols))
            lost += 1
        else:
            reward = np.sum(game[-1])
            if np.sum(game[-1]) == game.shape[1]:
                scored += 1 
                game[1:] = game[0:-1]
                game[0] = 0
 
        pos, lower_pos = create_piece(game)

    if iteration % 1000 == 0:
        print(f"--------------------\n")
        print(f"iteration: {iteration}")
        print(f"epsilon: {epsilon}")
        print(f"action: {action}")
        print(f"reward: {reward}")
        print(f"lost: {lost}")
        print(f"scored: {scored}")
        print(f"loss (last iteration): {loss}")
        print(f"\n{game}\n--------------------\n")

    memory.push(numpy_to_tensor(old_state), action,
                numpy_to_tensor(game), reward)

    if update_iter == update_step:
        loss = update(loss_function, optimizer)
        update_iter = 0
    update_iter += 1

    if copy_weights_iter == copy_weights_step:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        copy_weights_iter = 0
    copy_weights_iter += 1
    
    iteration += 1
