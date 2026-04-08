import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 mixing_schedule=[0.5, 2000, 4000], 
                 min_policy_std=None,
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 k_value=0.0,
                 cost_value_loss_coef=1.0,
                 cost_viol_loss_coef=1.0
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.min_policy_std = torch.tensor(min_policy_std, device=self.device)
        self.k_value = k_value
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef

        self.mixing_schedule = mixing_schedule
        self.counter = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, cost_shape, cost_d_values):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, cost_shape, cost_d_values, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, hist_encoding=False):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, leg_rewards, arm_rewards, costs, dones, infos):
        self.transition.rewards = torch.stack([leg_rewards.clone(), arm_rewards.clone()], dim=-1)
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
        
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs):
        last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)

    def compute_surrogate_loss(self, value_mixing_ratio, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        mixing_advantages_batch = torch.zeros_like(advantages_batch)
        mixing_advantages_batch[..., 0] = advantages_batch[..., 0] + value_mixing_ratio * advantages_batch[..., 1]
        mixing_advantages_batch[..., 1] = advantages_batch[..., 1] + value_mixing_ratio * advantages_batch[..., 0]
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
        surrogate = - mixing_advantages_batch * ratio
        surrogate_clipped = - mixing_advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss
    
    def compute_cost_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = cost_advantages_batch * ratio[:, 0].view(-1,1)
        surrogate_clipped = cost_advantages_batch * torch.clamp(ratio[:, 0].view(-1,1), 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(0)
        return surrogate_loss
    
    def compute_value_loss(self, target_values_batch, value_batch, returns_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss
    
    def update_k_value(self, i):
        self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004**i))
        return self.k_value
    
    def compute_viol(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch, cost_volation_batch):

        cost_surrogate_loss = self.compute_cost_surrogate_loss(actions_log_prob_batch,
                                                            old_actions_log_prob_batch,
                                                            cost_advantages_batch)
        cost_volation_loss = cost_volation_batch.mean()
        cost_loss = cost_surrogate_loss + cost_volation_loss
        cost_loss = torch.sum(self.k_value * F.relu(cost_loss))

        return cost_loss

    def update(self):
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_viol_loss = 0
        mean_surrogate_loss = 0
        mean_priv_reg_loss = 0
        value_mixing_ratio = self.get_value_mixing_ratio()
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, \
            target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch in generator:

                self.actor_critic.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                surrogate_loss = self.compute_surrogate_loss(value_mixing_ratio, 
                                                             actions_log_prob_batch, 
                                                             old_actions_log_prob_batch, 
                                                             advantages_batch)
                # Cost violation
                viol_loss = self.compute_viol(actions_log_prob_batch,
                                            old_actions_log_prob_batch,
                                            cost_advantages_batch,
                                            cost_violation_batch)
                # Value function loss
                value_loss = self.compute_value_loss(target_values_batch,
                                                    value_batch,
                                                    returns_batch)
                # Cost value function loss
                cost_value_loss = self.compute_value_loss(target_cost_values_batch,
                                                        cost_value_batch,
                                                        cost_returns_batch)


                loss = surrogate_loss + self.cost_viol_loss_coef * viol_loss \
                       + self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss \
                       - self.entropy_coef * entropy_batch.mean() \
                       + priv_reg_coef * priv_reg_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_priv_reg_loss += priv_reg_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_viol_loss += viol_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_viol_loss /= num_updates

        self.storage.clear()
        self.update_counter()
        self.enforce_min_std()

        return mean_value_loss, mean_surrogate_loss, value_mixing_ratio, mean_priv_reg_loss, priv_reg_coef, mean_cost_value_loss, mean_viol_loss
    
    def update_dagger(self):
        mean_hist_latent_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, \
            target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates

        self.storage.clear()
        self.update_counter()

        return mean_hist_latent_loss

    def enforce_min_std(self):
        current_std = self.actor_critic.std.detach()
        new_std = torch.max(current_std, self.min_policy_std).detach()
        self.actor_critic.std.data = new_std
    
    def update_counter(self):
        self.counter += 1
    
    def get_value_mixing_ratio(self):
        return min(max((self.counter - self.mixing_schedule[1]) / self.mixing_schedule[2], 0), 1) * self.mixing_schedule[0]