import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment.

    Parameters:
        state_size (int):   dimension of each state
        action_size (int):  dimension of each action
        seed (int):         random seed
        params:
            model_type:         DQN or DDQN
            buffer_size:        replay buffer size
            batch_size:         minibatch size
            alpha:              prioritization level (ALPHA=0 is uniform sampling so no prioritization)
            beta_start:               controls how much IS weightings affect learning (beta=0 at start moving to 1 towards the end )
            beta_increments:    the number of intervals beta will increase to get to 1.0
            gamma:              discount factor
            target_tau:         for soft update of target parameters
            learning_rate:      learning rate 
            update_rate:        how often to update the network

    """

    def __init__(self, state_size, action_size, hidden_layers, seed, params, filename=None):
        
        """Initialize an Agent object.
        
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            model_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
            buffer_size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            seed (int): random seed for initializing training point.
        """
        
        self.model_type = params.model_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(params.buffer_size)
        self.batch_size = params.batch_size
        self.alpha = params.alpha
        self.beta_end = 1.0
        self.beta = params.beta_start
        self.beta_increment = (self.beta_end - params.beta_start)/params.beta_increments
        self.gamma = params.gamma
        self.learn_rate = params.learning_rate
        self.tau = params.target_tau
        self.update_rate = params.update_rate
        self.seed = random.seed(seed)

        # Q-Network

        if filename:
            weights = torch.load(filename)
            self.qnetwork_local.load_state_dict(weights)
            #self.qnetwork_target.load_state_dict(weights)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, seed, params).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, seed, params).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learn_rate)

            
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        
    # ================================================================================ #
    # STEP() method
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                if (self.model_type == 'PERDDQN'):
                    experiences = self.memory.Prioritysample(self.alpha, self.beta)
                else:
                    experiences = self.memory.sample()

                self.learn(experiences, self.gamma)

        if done:
            self.increase_beta()

            
    # ================================================================================ #
    # ACT() method
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        # "unsqueeze" set the batch_size dim which is one here
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)

        
        
    # ================================================================================ #
    # LEARN() method
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        

        if (self.model_type == 'PERDDQN'):
            #Double DQN with Priority Replay
            #************************
            
            states, actions, rewards, next_states, dones, weights, indices = experiences

            # Get Q values from current observations (s, a) using model nextwork
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            
            # Get max action from local model
            local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(), 1, local_max_actions)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Compute loss
            loss  = (Q_expected - Q_targets).pow(2) * weights
            prios = loss + 1e-5
            loss  = loss.mean()            
        
        elif (self.model_type == 'DDQN'):

            #Double DQN
            #************************
            
            states, actions, rewards, next_states, dones = experiences

            # Get Q values from current observations (s, a) using model nextwork
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            
            # Get max action from local model
            local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(), 1, local_max_actions)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            loss = F.mse_loss(Q_expected, Q_targets)
            
        else:
            #Regular (Vanilla) DQN
            #************************

            states, actions, rewards, next_states, dones = experiences

            # Get Q values from current observations (s, a) using model nextwork
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Get max Q values for (s',a') from target model
            Q_target_values = self.qnetwork_target(next_states).detach()
            Q_targets_next = Q_target_values.max(1)[0].unsqueeze(1)        
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Compute loss (error)
            loss = F.mse_loss(Q_expected, Q_targets)


        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.model_type == 'PERDDQN'):
            # Update priorities based on td error
            self.memory.update_priorities(indices.squeeze().to('cpu').data.numpy(), prios.squeeze().to('cpu').data.numpy())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

        
    # ================================================================================ #
    # SOFT_UPDATE() method
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def increase_beta(self):
        if self.beta < self.beta_end:
            self.beta = min(self.beta_end, self.beta + self.beta_increment)


            

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done, max_priority)
        self.memory.append(e)
    
    def sample(self):

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def Prioritysample(self, alpha, beta):
        """ sample recall buffer based on experience priority """

        # Probabilities associated with each entry in memory
        priorities = np.array([sample.priority for sample in self.memory])
        probs  = priorities ** alpha
        probs /= probs.sum()
        
        # Get indices
        indices = np.random.choice(len(self.memory), self.batch_size, replace = False, p=probs)
        
        # Associated experiences
        experiences = [self.memory[idx] for idx in indices]    

        # Importance sampling weights
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)        
        indices = torch.from_numpy(np.vstack(indices)).long().to(device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, priorities):
        for i, idx in enumerate(indices):
            # A tuple is immutable so need to use "_replace" method to update it - might replace the named tuple by a dict
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])
    # -----------------------------------------------------------------


    
    
