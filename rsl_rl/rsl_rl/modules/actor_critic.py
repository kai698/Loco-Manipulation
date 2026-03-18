import torch
import torch.nn as nn
from torch.distributions import Normal

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):

        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        channel_size = 10

        self.encoder = nn.Sequential(nn.Linear(input_size, 3 * channel_size), self.activation_fn)

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, 
                    nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size), self.activation_fn)

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
    
class Actor(nn.Module):
    def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, \
                    leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                    num_leg_actions, num_arm_actions, \
                    num_priv, num_hist, num_prop, priv_encoder_dims):
        super(Actor, self).__init__()

        # Priv Encoder
        self.priv_encoder = mlp_backbone(num_priv, priv_encoder_dims, activation)
        priv_encoder_output_dim = priv_encoder_dims[-1]

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.history_encoder = StateHistoryEncoder(activation, mlp_input_dim_a, num_hist, priv_encoder_output_dim)

        # Policy
        self.actor_backbone = mlp_backbone(mlp_input_dim_a + priv_encoder_output_dim, actor_hidden_dims, activation)
        actor_backbone_output_dim = actor_hidden_dims[-1]

        self.actor_leg_control_head = mlp(actor_backbone_output_dim, leg_control_head_hidden_dims, num_leg_actions, activation)
        self.actor_arm_control_head = mlp(actor_backbone_output_dim, arm_control_head_hidden_dims, num_arm_actions, activation)
    
    def forward(self, obs, hist_encoding: bool = False):
        obs_prop = obs[:, :self.num_prop]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        leg_output = self.actor_leg_control_head(backbone_output)
        arm_output = self.actor_arm_control_head(backbone_output)
        return torch.cat([leg_output, arm_output], dim=-1)
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop: self.num_prop + self.num_priv]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
class Critic(nn.Module):
    def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation, \
                    leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                    num_priv, num_hist, num_prop):
        super(Critic, self).__init__()

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        # Value
        self.critic_backbone = mlp_backbone(mlp_input_dim_c, critic_hidden_dims, activation)
        critic_backbone_output_dim = critic_hidden_dims[-1]

        self.critic_leg_control_head = mlp(critic_backbone_output_dim, leg_control_head_hidden_dims, 1, activation)
        self.critic_arm_control_head = mlp(critic_backbone_output_dim, arm_control_head_hidden_dims, 1, activation)
    
    def forward(self, obs):
        prop_and_priv = obs[:, :self.num_prop + self.num_priv]
        backbone_output = self.critic_backbone(prop_and_priv)
        leg_output = self.critic_leg_control_head(backbone_output)
        arm_output = self.critic_arm_control_head(backbone_output)
        return torch.cat([leg_output, arm_output], dim=-1)

class ActorCritic(nn.Module):

    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        priv_encoder_dims=[64, 20],
                        activation='elu',
                        init_std=1,
                        **kwargs):
        # if kwargs:
        #     print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        leg_control_head_hidden_dims = kwargs['leg_control_head_hidden_dims']
        arm_control_head_hidden_dims = kwargs['arm_control_head_hidden_dims']
        self.num_leg_actions = kwargs['num_leg_actions']
        self.num_arm_actions = kwargs['num_arm_actions']
        num_priv = kwargs['num_priv']
        num_hist = kwargs['num_hist']
        num_prop = kwargs['num_prop']

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.actor = Actor(mlp_input_dim_a, actor_hidden_dims, activation, \
                           leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                           self.num_leg_actions, self.num_arm_actions, \
                           num_priv, num_hist, num_prop, priv_encoder_dims)

        self.critic = Critic(mlp_input_dim_c + num_priv, critic_hidden_dims, activation, \
                             leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                             num_priv, num_hist, num_prop)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(torch.tensor(init_std))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        entropy = self.distribution.entropy()
        leg_entropy_sum = entropy[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_entropy_sum = entropy[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_entropy_sum, arm_entropy_sum], dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        leg_log_prob_sum = log_prob[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_log_prob_sum = log_prob[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_log_prob_sum, arm_log_prob_sum], dim=-1)

    def act_inference(self, observations, hist_encoding=False):
        actions_mean = self.actor(observations, hist_encoding)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def mlp_backbone(input_dim, hidden_dims, activation):
    """MLP backbone: all hidden layers each followed by activation, no output projection."""
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for l in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)
    return nn.Sequential(*layers)

def mlp(input_dim, hidden_dims, output_dim, activation):
    """MLP: hidden layers with activation, final linear layer without activation."""
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)
    return nn.Sequential(*layers)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None