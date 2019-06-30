import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
GRACE_PERIOD = int(1e3)  # withold learning up to this
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
UPDATE_FREQ = 10  # update frequency
WEIGHT_DECAY = 1e-2  # L2 weight decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def final_layer_init(layer):
    nn.init.uniform_(layer.weight.data, -3e-3, 3e-3)
    return layer


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight.data, -lim, lim)
    return layer


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, max_action, fc_units=(400, 300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        #        self.layers.append(nn.BatchNorm1d(state_size))
        for i, (inp, outp) in enumerate(zip((state_size,) + fc_units, fc_units + (action_size,))):
            if i < len(fc_units):
                self.layers.append(hidden_init(nn.Linear(inp, outp)))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(final_layer_init(nn.Linear(inp, outp)))

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        for layer in self.layers:
            state = layer(state)
        return torch.tanh(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, max_action, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.norm = nn.BatchNorm1d(state_size)
        self.fcs1 = hidden_init(nn.Linear(state_size, fcs1_units))
        self.fc2 = hidden_init(nn.Linear(fcs1_units + action_size, fc2_units))
        self.fc3 = hidden_init(nn.Linear(fc2_units, fc3_units))
        self.fc4 = final_layer_init(nn.Linear(fc3_units, 1))

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.norm(state)
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.noise = OrnsteinUhlenbeckProcess(action_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        for _ in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename), map_location=device))


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, memory=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.updated = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc_units=(state_size * 10, action_size * 40)).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc_units=(state_size * 10, action_size * 40)).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OrnsteinUhlenbeckProcess(action_size)

        # Replay memory
        self.memory = memory or ReplayBuffer(random_seed)

        if random_seed is not None:
            random.seed(random_seed)

    def step(self, n, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > max(GRACE_PERIOD, BATCH_SIZE):
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, n % UPDATE_FREQ == 0 and n > 0)

    def select_action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        if len(self.memory) > GRACE_PERIOD:
            self.updated += 1
            state = torch.from_numpy(state).float().to(device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                action += self.noise.sample()
        else:
            action = [[random.uniform(-1, 1) for _ in range(self.action_size)] for _ in range(self.num_agents)]
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma, update):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).detach()

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if update:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1, dt=1e-2, x0=None):
        self.theta = theta  # time constant, 1 / tau
        self.sigma = sigma  # standard deviation
        self.mu = mu  # mean
        self.dt = dt  # time step
        self.x0 = x0  # initial values
        self.size = size  # number of samples
        self.a = theta * dt
        self.b = sigma * np.sqrt(2.0 * theta * dt)
        self.reset()

    def reset(self):
        self.x_prev = copy.copy(self.x0) if self.x0 is not None else np.ones(self.size) * self.mu

    def sample(self):
        dx = self.a * (self.mu - self.x_prev) + self.b * np.random.randn(self.size)
        self.x_prev += dx
        return self.x_prev
