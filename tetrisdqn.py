import torch
from torch import nn
from torch import optim
from collections import namedtuple, deque
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

game_rows = 4 * 4
game_cols = 4 * 1
middle = math.floor(game_cols / 2)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
        self.feed = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (game_rows // 4) * (game_cols // 4), 256),
            nn.Tanh(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.feed(x)


policy_net = DqnNet().to(device)
print(f"policy_net on cuda: {next(policy_net.parameters()).is_cuda}")
target_net = DqnNet().to(device)
print(f"target_net on cuda: {next(target_net.parameters()).is_cuda}")
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
loss_function = nn.SmoothL1Loss()  # Huber Loss
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
memory = ExperienceReplay(100000)


def create_piece(state):
    if random.random() > 0.5:
        state[0, 0, 0, middle] = 1
    else:
        state[0, 0, 0, middle] = 1
        state[0, 0, 0, middle - 1] = 1
        state[0, 0, 0, middle + 1] = 1


def turn_piece(state):
    if state[0, 0].nonzero().numel() / 2 == 3:
        a, b, c = state[0, 0].nonzero()[:, 0]
        d, e, f = state[0, 0].nonzero()[:, 1]
        if a == b == c:
            if not piece_landed(state):
                state[0, 0, b, d] = 0
                state[0, 0, b, f] = 0
                state[0, 0, b - 1, e] = 1
                state[0, 0, b + 1, e] = 1
        elif d == e == f:
            if (not (0 in state[0, 0].nonzero()[:, 1])) and (
                not (game_cols - 1 in state[0, 0].nonzero()[:, 1])
            ):
                if (
                    state[0, 1, a, e - 1] != 1
                    and state[0, 1, a, e + 1] != 1
                    and state[0, 1, b, e - 1] != 1
                    and state[0, 1, b, e + 1] != 1
                    and state[0, 1, c, e - 1] != 1
                    and state[0, 1, c, e + 1] != 1
                ):
                    state[0, 0, a, e] = 0
                    state[0, 0, c, e] = 0
                    state[0, 0, b, e - 1] = 1
                    state[0, 0, b, e + 1] = 1


def move_piece(state, action):
    state[0, 0] = torch.roll(state[0, 0], 1, 0)
    if action == 1:
        if not (0 in state[0, 0].nonzero()[:, 1]):
            state[0, 0] = torch.roll(state[0, 0], -1, 1)
            if 2 in torch.sum(state, dim=1):
                state[0, 0] = torch.roll(state[0, 0], 1, 1)
    if action == 2:
        if not (game_cols - 1 in state[0, 0].nonzero()[:, 1]):
            state[0, 0] = torch.roll(state[0, 0], 1, 1)
            if 2 in torch.sum(state, dim=1):
                state[0, 0] = torch.roll(state[0, 0], -1, 1)
    if action == 3:
        turn_piece(state)


def piece_landed(state):
    landed = False
    state[0, 0] = torch.roll(state[0, 0], 1, 0)
    if 2 in torch.sum(state, dim=1):
        landed = True
    state[0, 0] = torch.roll(state[0, 0], -1, 0)
    if game_rows - 1 in torch.index_select(
        state[0, 0].nonzero(), 1, torch.tensor([0]).to(device)
    ):
        landed = True
    return landed


def transfer_piece(state):
    state[0, 1] = torch.sum(state, dim=1)
    state[0, 0] = 0


def game_over():
    return True if torch.sum(state[0, 1, 1, :]).item() > 0 else False


def restart_game():
    state = torch.zeros(1, 2, game_rows, game_cols)
    state = state.to(device)
    create_piece(state)
    return state


def score(state):
    scored_rows = 0
    for row in range(game_rows):
        if torch.sum(state[0, 1, row, :]) == game_cols:
            scored_rows += 1
            tmp = torch.clone(state)
            state[0, 1, 1 : row + 1, :] = tmp[0, 1, 0:row, :]
    return scored_rows


def evaluate(state):
    reward = 0
    # give reward according to which row the piece is. The further the piece progressed, the higher the reward
    reward += torch.max(state[0, 0].nonzero()[:, 0]).item() / 5
    # give reward according to how many pieces are below the currently moving piece. The more pieces the higher the negative reward
    curr_piece_cols = state[0, 0].nonzero()[:, 1]
    reward -= (
        torch.sum(torch.unsqueeze(torch.sum(state[0, 1], 0), 0)[:, curr_piece_cols])
        / curr_piece_cols.numel()
    ).item() / 3
    return reward


epsilon = 1
epsilon_end = 0.05
decay = 0.999
discount = 0.9
random.seed()


def select_action(state):
    global epsilon
    rand = random.random()
    if rand <= epsilon:
        action = random.randint(0, 3)
        if epsilon > epsilon_end:
            epsilon *= decay
    else:
        with torch.no_grad():
            predictions = policy_net(state)
        action = torch.argmax(predictions).item()
    return action


def update(loss_function, optimizer):
    if len(memory) < 64:
        return
    target = []
    prediction = []
    replays = memory.sample(64)
    for replay in replays:
        prediction.append(
            torch.unsqueeze(policy_net(replay.state)[0, replay.action], 0)
        )
        target.append(
            torch.unsqueeze(
                replay.reward + discount * torch.max(target_net(replay.next_state)), 0
            )
        )  # ignore that a final state exists
        # if replay.reward >= 0:
        #   target.append(torch.unsqueeze(replay.reward + discount * torch.max(target_net(replay.next_state)), 0))
        # else:
        #   target.append(torch.unsqueeze(replay.reward + (discount*0.25) * torch.max(target_net(replay.next_state)), 0))
    prediction = torch.cat(prediction)
    target = torch.cat(target)
    loss = loss_function(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


update_iter = 0
update_step = 64
copy_weights_iter = 0
copy_weights_step = update_step * 10

lost = 0
scored = 0
iteration = 0
loss = 0

state = restart_game()

for _ in range(1000000):
    reward = 0
    old_state = torch.clone(state)
    action = select_action(state)
    move_piece(state, action)
    reward += evaluate(state)
    if piece_landed(state):
        transfer_piece(state)
        if game_over():
            reward = -(game_rows - 1)
            lost += 1
            state = restart_game()
        else:
            scored_rows = score(state)
            scored += scored_rows
            reward += scored_rows * game_cols
            create_piece(state)

    memory.push(old_state, action, torch.clone(state), reward)

    if iteration % 1000 == 0:
        print(
            f"""
    iteration: {iteration}
    epsilon: {epsilon}
    action: {action}
    reward: {reward}
    loss (last update): {loss}
    lost: {lost}
    scored: {scored}
    {state}"""
        )

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
