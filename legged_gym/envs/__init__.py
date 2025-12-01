from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.go2w.go2w_config import Go2wCfg, Go2wCfgPPO
from legged_gym.envs.go2w.go2w import Go2w

task_registry.register( "go2w", Go2w, Go2wCfg(), Go2wCfgPPO())
