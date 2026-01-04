import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2w_piper.go2w_piper_config import Go2wPiperCfg, Go2wPiperCfgPPO
from rsl_rl.modules.actor_critic import StateHistoryEncoder

import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, \
                    leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                    num_leg_actions, num_arm_actions, \
                    num_priv, num_hist, num_prop, priv_encoder_dims, output_tanh=False):
        super().__init__()

        # Priv Encoder
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        self.history_encoder = StateHistoryEncoder(activation, mlp_input_dim_a, num_hist, priv_encoder_output_dim)

        # Policy
        if len(actor_hidden_dims) > 0:
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a + priv_encoder_output_dim, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims) - 1):
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
            self.actor_backbone = nn.Sequential(*actor_layers)
            actor_backbone_output_dim = actor_hidden_dims[-1]
        else:
            self.actor_backbone = nn.Identity()
            actor_backbone_output_dim = mlp_input_dim_a + priv_encoder_output_dim

        actor_leg_layers = []
        actor_leg_layers.append(nn.Linear(actor_backbone_output_dim, leg_control_head_hidden_dims[0]))
        actor_leg_layers.append(activation)
        for l in range(len(leg_control_head_hidden_dims)):
            if l == len(leg_control_head_hidden_dims) - 1:
                actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], num_leg_actions))
                if output_tanh:
                    actor_leg_layers.append(nn.Tanh())
            else:
                actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                actor_leg_layers.append(activation)
        self.actor_leg_control_head = nn.Sequential(*actor_leg_layers)

        actor_arm_layers = []
        actor_arm_layers.append(nn.Linear(actor_backbone_output_dim, arm_control_head_hidden_dims[0]))
        actor_arm_layers.append(activation)
        for l in range(len(arm_control_head_hidden_dims)):
            if l == len(arm_control_head_hidden_dims) - 1:
                actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], num_arm_actions))
                if output_tanh:
                    actor_arm_layers.append(nn.Tanh())
            else:
                actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                actor_arm_layers.append(activation)
        self.actor_arm_control_head = nn.Sequential(*actor_arm_layers)
    
    def forward(self, obs_prop_and_latent):
        backbone_input = obs_prop_and_latent
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

def save(cfg: Go2wPiperCfg, train_cfg: Go2wPiperCfgPPO):

    actor = Actor(
        cfg.env.num_proprio,
        train_cfg.policy.actor_hidden_dims,
        nn.ELU(),
        train_cfg.policy.leg_control_head_hidden_dims,
        train_cfg.policy.arm_control_head_hidden_dims,
        train_cfg.policy.num_leg_actions,
        train_cfg.policy.num_arm_actions,
        cfg.env.num_priv,
        cfg.env.history_len, 
        cfg.env.num_proprio,
        train_cfg.policy.priv_encoder_dims,
        train_cfg.policy.output_tanh
    )
    actor.eval()
    actor.cpu()

    export_root = os.path.join(
                            LEGGED_GYM_ROOT_DIR,
                            "logs",
                            train_cfg.runner.experiment_name,
                            "exported",
                        )
    load_path = os.path.join(export_root, "model_actor.pt")
    print(f"Loading exported actor model from: {load_path}")

    save_root = os.path.join(export_root, "traced")
    os.makedirs(save_root, exist_ok=True)
    actor.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    # Save the traced actor
    dummy_actor_input = torch.zeros(1, train_cfg.policy.priv_encoder_dims[-1] + cfg.env.num_proprio)
    with torch.no_grad():
        actor(dummy_actor_input)
        traced_actor = torch.jit.trace(actor, dummy_actor_input)
    save_path = os.path.join(save_root, "traced_actor.pt")
    traced_actor.save(save_path)
    print(f"Saved traced actor model to: {save_path}")

    # Save the traced history encoder
    dummy_hist_input = torch.zeros(1, cfg.env.history_len * cfg.env.num_proprio)
    with torch.no_grad():
        actor.history_encoder(dummy_hist_input)
        traced_hist_encoder = torch.jit.trace(actor.history_encoder, dummy_hist_input)
    save_path = os.path.join(save_root, "traced_hist_encoder.pt")
    traced_hist_encoder.save(save_path)
    print(f"Saved traced history encoder model to: {save_path}")

if __name__ == '__main__':
    save(Go2wPiperCfg, Go2wPiperCfgPPO)
