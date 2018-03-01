import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import vizdoom
import random
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import time

plt.ion()

'''
================================================================================
'''

game = vizdoom.DoomGame()

path = 'C:/Users/georg/Documents/GitHub/Vizdoom-MemQN/scenarios/' + \
       'my_way_home.wad'
game.set_doom_scenario_path(path)
game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
game.set_screen_format(vizdoom.ScreenFormat.RGB24)  # [0, 255]^3 for each pixel

# Get position data
game.add_available_game_variable(vizdoom.GameVariable.POSITION_X)
game.add_available_game_variable(vizdoom.GameVariable.POSITION_Y)

# Don't render what's not nessecary
game.set_render_hud(False)
game.set_render_minimal_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# Add buttons
game.add_available_button(vizdoom.Button.MOVE_LEFT)
game.add_available_button(vizdoom.Button.MOVE_RIGHT)
game.add_available_button(vizdoom.Button.MOVE_FORWARD)
game.add_available_button(vizdoom.Button.TURN_LEFT)
game.add_available_button(vizdoom.Button.TURN_RIGHT)

# Only one action at a time
actions = [[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]]

# Episode and display settings
game.set_episode_timeout(2100)
game.set_episode_start_time(0)
game.set_living_reward(-0.0001)
game.set_mode(vizdoom.Mode.PLAYER)
game.set_window_visible(True)

game.init()


'''
================================================================================
'''

# Use cuda if available
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition', ('state',
                                       'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, im_height, im_width, hidden_size, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.encoding_size = int(im_height * im_width / 32)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.encoding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.encoding_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 4000 max for GTX 1060
replay_mem = ReplayMemory(10000)
model = DQN(240, 320, hidden_size=256, num_actions=len(actions))
target_model = DQN(240, 320, hidden_size=256, num_actions=len(actions))
if use_cuda:
    model.cuda()
    target_model.cuda()
optimizer = optim.Adam(model.parameters())

steps_done = 0
# e is for epsilon
e_start = 0.9
e_end = 0.1
e_decay = 10000


def takeAction(actions, state):
    global steps_done
    # threshold = end + (start - end) * e^(-steps_done / decay)
    e_threshold = e_end + (e_start - e_end) * \
        math.exp(-1 * steps_done / e_decay)
    e_threshold = 0

    steps_done += 1

    if random.random() <= e_threshold:  # epsilon greedy policy
        return random.choice(actions)
    else:
        action_tensor = model(Variable(state, volatile=True)).data.max(1)[1]
        return actions[action_tensor.cpu().numpy()[0]]  # all to get a number


last_sync = 0
sync_every = 1000


# Calculates cost and then optimizes
def train(batch_size, gamma):
    global last_sync

    if len(replay_mem) < batch_size:  # escape if not enough transitions
        return

    transitions = replay_mem.sample(batch_size)  # Get list of transitions
    batch = Transition(*zip(*transitions))  # Turn into tuple of lists

    # Wrap in variables
    state_batch = Variable(torch.cat(batch.state), requires_grad=False)
    # Turn action_batch into vector of indices
    action_batch = torch.cat(batch.action).max(1)[1]
    action_batch = Variable(action_batch, volatile=True, requires_grad=False)
    next_state_batch = Variable(
        torch.cat(batch.next_state), volatile=True, requires_grad=False)
    reward_batch = Variable(torch.cat(batch.reward),
                            volatile=True, requires_grad=False)

    # Compute Q(s, a)
    # https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
    Q = model(state_batch).gather(1, action_batch.view(-1, 1))
    # Occasionally sync target_model and model
    if last_sync >= sync_every:
        target_model.load_state_dict(model.state_dict())
    last_sync += 1
    # Compute max_a Q(s', a) using target network
    max_Q = target_model(next_state_batch).max(1)[0]

    # Calculate loss using Q*(s, a) = E(r + y max_a' Q*(s', a'))
    target_Q = reward_batch + gamma * max_Q
    # Huber loss
    loss = F.smooth_l1_loss(Q, target_Q)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.grad.data.clamp_(-1, 1)
    optimizer.step()


def plot(eps_losses, delay):
    plt.figure(1)
    plt.clf()
    losses_torch = torch.FloatTensor(eps_losses)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(losses_torch.numpy())
    # Take 100 episode averages and plot them too
    if len(losses_torch) >= 100:
        means = losses_torch.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    # Flashes plot for 2 seconds (I couldn't get it to work with interactive
    # mode)
    plt.show()
    plt.pause(delay)
    plt.close(1)


'''
===============================================================================
'''


def timeSince(since):
    return time.time() - since


# Load/save control
load_model = False
save_model = True
save_path = 'C:/Users/georg/Documents/GitHub/Vizdoom-MemQN/dqn.json'

# Train control
while_train = True

# Load model if desired
if load_model:
    model.load_state_dict(torch.load(save_path))
    target_model.load_state_dict(torch.load(save_path))

# Init last screen to zeros
last_screen = torch.zeros(1, 3, 640, 480)

batch_size = 128
eps_losses = []
chkpt = 500  # save every chkpt episodes
N_eps = 10
n_eps = 0  # number of epsiodes so far
train_time = 60 * 60 * 24 * 2  # 2 days
show_step_stats = False  # prob not needed for a 2 day train

start = time.time()


def step():  # so there is less boiler plate code
    global last_screen
    print("Episode %d" % (n_eps + 1))

    game.new_episode()
    eps_loss = 0

    while not game.is_episode_finished():
        state = game.get_state()
        vars = state.game_variables
        screen = state.screen_buffer
        # Convert to C x H x W np array
        screen = screen.transpose((2, 0, 1))
        screen = torch.from_numpy(screen).type(FloatTensor)
        screen = screen.unsqueeze(0)  # batch_dim = 1

        a = takeAction(actions, screen)
        r = game.make_action(a)

        if state.number != 1:  # skip
            # Turn a and r into tensors before pushing
            # Unsqueeze needed for cat in train
            a = LongTensor(a).unsqueeze(0)
            replay_mem.push(last_screen, a, screen, Tensor([r]))

        last_screen = screen

        train(batch_size, 0.999)

        eps_loss += r

        if show_step_stats:  # prob not nessecary for a 2 day train
            print("State:", state.number)
            print("Variables:", vars)
            print("Reward:", r)  # easier to print r than Tensor([r])
            print("=====================")

    eps_losses.append(eps_loss)
    plot(eps_losses, 2)  # I couldn't get the graph to dynamically update

    # Save at checkpoint
    if save_model and n_eps % chkpt == 0:
        torch.save(model.state_dict(), save_path)


if while_train:
    while timeSince(start) <= train_time:
        step()
else:
    for n_eps in range(N_eps):
        step()

# Save model at end if desired
if save_model:
    torch.save(model.state_dict(), save_path)

plot(eps_losses, 60 * 60 * 24)  # plot stays up for a day

# For the sake of completeness
game.close()
plt.ioff()
