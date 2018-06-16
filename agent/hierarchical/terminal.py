import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import modules
except:
    import agent.hierarchical.modules as modules

from torch.autograd import Variable

class TerminalPolicy(nn.Module):
    def __init__(self, num_actions, num_timesteps, vocabulary_size):
        '''
        num_actions - number of low-level actions
        num_timesteps - number of frames for one state
        vocabulary_size - number of possible unique words in the instruction
        '''
        super(TerminalPolicy, self).__init__()

        self.num_actions = num_actions
        self.num_timesteps = num_timesteps

        self.vision = modules.VisualEncoder()
        self.language = modules.InstructionEncoder(vocabulary_size=vocabulary_size)
        self.fuser = modules.Fusion()
        self.time = modules.TimeEncoder()
        self.actions = modules.AugmentedPolicy(num_actions)
        self.value_function = nn.Linear(in_features=256, out_features=1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0001)

    def _value_function(self, time_encoded):
        '''
        input:
            time_encoded - a tensor of size (batch_size, 256)
        output:
            state_returns - a tensor of size (batch_size, 1)
        '''
        return self.value_function(time_encoded)
    
    def forward(self, frames, instructions):
        '''
        input:
            frames - a tensor of size (batch_size, num_timesteps, 3, 84, 84)
            instructions - a tensor of size (batch_size, seq_length)
        output:
            action_probs - a tensor of size (batch_size, num_actions)
        '''
        vision_encoding = self.vision.forward(frames)
        language_encoding = self.language.forward(instructions)
        fused = self.fuser.forward(vision_encoding, language_encoding)
        time_encoding = self.time.forward(fused)

        action_probs = self.actions.forward(time_encoding)
        return action_probs

    def train(self, frames, instructions, actions, returns, action_probs):
        '''
        Trains the policy using Advantage Actor-Critic algorithm.

        input:
            frames - a tensor of size (batch_size, num_timesteps, 3, 84, 84)
            instructions - a tensor of size (batch_size, seq_length)
            actions - a tensor of size (batch_size, 1)
            returns - a tensor of size (batch_size, 1)
            action_probs - a tensor of size (batch_size, num_actions)
        output:
            loss - calculated loss for the minibatch
        '''
        vision_encoding = self.vision.forward(frames)
        language_encoding = self.language.forward(instructions)
        fused = self.fuser.forward(vision_encoding, language_encoding)
        time_encoding = self.time.forward(fused)
        cur_action_probs = self.actions.forward(time_encoding)

        advantage = returns - self._value_function(time_encoding)
        importance_weight = (cur_action_probs / action_probs).gather(1, actions)
        cur_action_log_probs = cur_action_probs.log().gather(1, actions)

        loss = -(importance_weight * (cur_action_log_probs * advantage)).mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 1.0)
        self.optimizer.step()

        return loss

if __name__ == "__main__":
    # TESTING
    terminal_policy = TerminalPolicy(num_actions=8, num_timesteps=4, vocabulary_size=10)
    num_parameters = sum(p.numel() for p in terminal_policy.parameters() if p.requires_grad)
    print("Terminal policy, num of parameters: " + str(num_parameters))

    action_probs = terminal_policy.forward(
        Variable(torch.randn(10, 4, 3, 84, 84)),
        Variable(torch.LongTensor(10, 8).random_(0, 9))
    )
    assert(action_probs.size() == (10, 8))
    print("Terminal policy: Output size is verified.")