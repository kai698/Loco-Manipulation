from isaacgym import gymapi
import os

def main():
    # Initialize gym and simulator
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z  # Z-axis up
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)  # Gravity along -Z
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # Environment bounds
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    # Add grid ground
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # Normal along Z-axis
    gym.add_ground(sim, plane_params)

    # Load URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    asset_root = os.path.join(current_dir, "resources/robots/go2w_description/urdf")
    asset_file = "go2w_description.urdf"

    opt = gymapi.AssetOptions()
    opt.collapse_fixed_joints = True
    opt.fix_base_link = False
    opt.flip_visual_attachments = True
    opt.replace_cylinder_with_capsule = True
    opt.disable_gravity = False
    asset = gym.load_asset(sim, asset_root, asset_file, opt)

    # Set initial base pose
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.5)
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

    # Create robot actor
    gym.create_actor(env, asset, pose, "go2w", 0, 0)

    # Camera look at robot
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 2, 1), gymapi.Vec3(0, 0, 0))

    # Main loop
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()