import sys

from dl_scaman_checker.common import __version__
from .TP01 import pretty_wrapped, pretty_warn

from IPython import display
from IPython.display import HTML
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation


@pretty_wrapped
def check_install():
    return f"Install ok. Version is v{__version__}"

@pretty_wrapped
def create_video(env, policy, num_frames=100, preprocess=None):
    def animation_update(num):
        progress_bar.update(1)
        ax.clear()
        state = env.render("rgb_array")
        ax.imshow(state)
        state = state if preprocess is None else preprocess(state)
        action = policy(state)
        env.step(action)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    env.reset()
    for _ in range(np.random.randint(1, 10)):
        env.step(0)
    state, _, done, _ = env.step(env.action_space.sample())
    progress_bar = tqdm(total=num_frames)
    anim = animation.FuncAnimation(fig, animation_update, frames=num_frames, interval=50)
    anim = HTML(anim.to_html5_video())
    progress_bar.close()
    plt.close()
    return anim

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.full = False
        self.curr = 0

        self.states = torch.zeros((self.max_size, 84, 84), dtype=torch.uint8, device=device)
        self.rewards = torch.zeros((self.max_size, 1), device=device)
        self.actions = torch.zeros((self.max_size, 1), dtype=torch.uint8, device=device)
        self.terminals = torch.zeros((self.max_size, 1), dtype=torch.uint8, device=device)

    def store(self, state, action, reward, terminal):
        idx = self.curr % self.max_size

        self.states[idx] = (state * 255).to(torch.uint8).to(device)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = terminal

        self._increment()

    def size(self):
        return self.max_size if self.full else self.curr

    def _increment(self):
        if self.curr + 1 == self.max_size:
            self.full = True

        self.curr = (self.curr + 1) % self.max_size

    def _index(self, idx):
        if len(idx) == 0 or self.size() < idx.max():
            raise ValueError("Not enough elements in cache to sample {} elements".format(len(idx)))

        return (self.states[idx].to(device).to(torch.float32) / 255.), \
                self.actions[idx].to(device).to(torch.long), \
                self.rewards[idx].to(device), \
                self.terminals[idx].to(device).to(torch.int16)

    def _process_idx(self, idx):
        return idx % self.max_size

    def sample(self, N):
        if self.size() - 4 < N:
            raise ValueError("Not enough elements in cache to sample {} elements".format(N))

        idx = np.random.choice(self.size() - 4, N)
        idx = self._process_idx(idx)
        state_idx = self._process_idx((idx.reshape(-1, 1) + np.array([0, 1, 2, 3]).reshape(1, -1)).flatten())
        next_state_idx = self._process_idx((idx.reshape(-1, 1) + np.array([0, 1, 2, 3]).reshape(1, -1)).flatten() + 1)

        states = []
        next_states = []
        actions = []
        rewards = []
        terminals = []

        _, a, r, t = self._index(idx)
        s, _, _, _ = self._index(state_idx)
        ns, _, _, _ = self._index(next_state_idx)

        states.append(s)
        next_states.append(ns)
        actions.append(a)
        rewards.append(r)
        terminals.append(t)

        return torch.cat(states).reshape(-1, 4, states[0].shape[1], states[0].shape[2]), \
               torch.cat(next_states).reshape(-1, 4, states[0].shape[1], states[0].shape[2]), \
               torch.cat(actions), \
               torch.cat(rewards), \
               torch.cat(terminals)
