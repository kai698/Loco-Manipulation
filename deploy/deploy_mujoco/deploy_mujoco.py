import mujoco
import mujoco.viewer
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from arm_controllers import Robot, CartesianIKController

def main():
    # model path
    xml_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/robots/go2w_piper/scene.xml"

    # load robot model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    piper = Robot(dof=6, base_link='arm_base', end_link='link7', joints=[f'joint{x}' for x in range(1, 7)],
                  actuators=[f'actuator{x}' for x in range(1, 7)])
    controller = CartesianIKController(piper)
    target = np.array([0.4, 0.0, 0.3, 0.5, -0.5, -0.5, -0.5])

    controller.set_model(model)
    controller.set_data(data)

    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            ctrl = controller.step_controller(target)

            for c, actuator in zip(ctrl, piper.actuators):
                data.actuator(actuator).ctrl = c

            # refresh viewer
            viewer.sync() 

if __name__ == "__main__":
    main()