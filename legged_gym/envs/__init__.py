from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.go2w.go2w_flat_config import Go2wFlatCfg, Go2wFlatCfgPPO
from legged_gym.envs.go2w.go2w_rough_config import Go2wRoughCfg, Go2wRoughCfgPPO
from legged_gym.envs.go2w.go2w import Go2w

task_registry.register( "go2w_flat", Go2w, Go2wFlatCfg(), Go2wFlatCfgPPO())
task_registry.register( "go2w_rough", Go2w, Go2wRoughCfg(), Go2wRoughCfgPPO())