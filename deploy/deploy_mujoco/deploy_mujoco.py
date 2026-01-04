import mujoco
import mujoco.viewer
import time
from legged_gym import LEGGED_GYM_ROOT_DIR

def main():
    # model path
    xml_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/robots/go2w_piper/scene.xml"

    # load robot model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            # refresh viewer
            viewer.sync() 

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()