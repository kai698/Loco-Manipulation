import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2w_piper.go2w_piper_config import Go2wPiperCfg, Go2wPiperCfgPPO
from rsl_rl.modules.actor_critic import Actor, get_activation

import torch

def save(cfg: Go2wPiperCfg, train_cfg: Go2wPiperCfgPPO):

    actor = Actor(
        cfg.env.num_proprio,
        train_cfg.policy.actor_hidden_dims,
        get_activation(train_cfg.policy.activation),
        train_cfg.policy.leg_control_head_hidden_dims,
        train_cfg.policy.arm_control_head_hidden_dims,
        train_cfg.policy.num_leg_actions,
        train_cfg.policy.num_arm_actions,
        cfg.env.num_priv,
        cfg.env.history_len, 
        cfg.env.num_proprio,
        train_cfg.policy.priv_encoder_dims)
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
