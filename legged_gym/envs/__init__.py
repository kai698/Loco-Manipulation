from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.go2w_piper.go2w_piper_config import Go2wPiperCfg, Go2wPiperCfgPPO
from legged_gym.envs.go2w_piper.go2w_piper import Go2wPiper

task_registry.register( "go2w_piper", Go2wPiper, Go2wPiperCfg(), Go2wPiperCfgPPO())