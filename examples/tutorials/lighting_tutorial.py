# [setup]
import math
import os

import magnum as mn
import numpy as np
from matplotlib import pyplot as plt

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils.common import quat_from_angle_axis

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../../data")
output_path = os.path.join(dir_path, "lighting_tutorial_output/")

save_index = 0


def show_img(data, save):
    plt.figure(figsize=(12, 12))
    plt.imshow(data, interpolation="nearest")
    plt.axis("off")
    plt.show(block=False)
    if save:
        global save_index
        plt.savefig(
            output_path + str(save_index) + ".jpg",
            bbox_inches="tight",
            pad_inches=0,
            quality=50,
        )
        save_index += 1
    plt.pause(1)


def get_obs(sim, show, save):
    obs = sim.get_sensor_observations()["rgba_camera"]
    if show:
        show_img(obs, save)
    return obs


def remove_all_objects(sim):
    for id_ in sim.get_existing_object_ids():
        sim.remove_object(id_)


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [5.0, 0.0, 1.0]
    agent_state.rotation = quat_from_angle_axis(
        math.radians(70), np.array([0, 1.0, 0])
    ) * quat_from_angle_axis(math.radians(-20), np.array([1.0, 0, 0]))
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    backend_cfg.enable_physics = True

    # agent configuration
    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [1080, 960]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


# [/setup]

# This is wrapped such that it can be added to a unit test
def main(show_imgs=True, save_imgs=False):
    if save_imgs and not os.path.exists(output_path):
        os.mkdir(output_path)

    # [default scene lighting]

    # create the simulator and render flat shaded scene
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)

    # [scene swap shader]

    # close the simulator and re-initialize with DEFAULT_LIGHTING_KEY:
    sim.close()
    cfg = make_configuration()
    cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)

    # create and register new light setup:
    my_scene_lighting_setup = [
        LightInfo(vector=[0.0, 2.0, 0.6, 0.0], model=LightPositionModel.Global)
    ]
    sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")

    # reconfigure with custom key:
    new_cfg = make_configuration()
    new_cfg.sim_cfg.scene_light_setup = "my_scene_lighting"
    sim.reconfigure(new_cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)

    # [/scene]

    # reset to default scene shading
    sim.close()
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)  # noqa: F841

    # [example 2]

    # get the physics object attributes manager
    obj_templates_mgr = sim.get_object_template_manager()

    # load some object templates from configuration files
    sphere_template_id = obj_templates_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects/sphere"))
    )[0]
    chair_template_id = obj_templates_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects/chair"))
    )[0]

    id_1 = sim.add_object(sphere_template_id)
    sim.set_translation([3.2, 0.23, 0.03], id_1)

    get_obs(sim, show_imgs, save_imgs)

    # [/example 2]

    # [example 3]

    # create a custom light setup
    my_default_lighting = [
        LightInfo(vector=[2.0, 2.0, 1.0, 0.0], model=LightPositionModel.Camera)
    ]
    # overwrite the default DEFAULT_LIGHTING_KEY light setup
    sim.set_light_setup(my_default_lighting)

    get_obs(sim, show_imgs, save_imgs)

    # [/example 3]

    # [example 4]
    id_2 = sim.add_object(chair_template_id)
    sim.set_rotation(mn.Quaternion.rotation(mn.Deg(-115), mn.Vector3.y_axis()), id_2)
    sim.set_translation([3.06, 0.47, 1.15], id_2)

    get_obs(sim, show_imgs, save_imgs)

    # [/example 4]

    # [example 5]
    light_setup_2 = [
        LightInfo(
            vector=[2.0, 1.5, 5.0, 1.0],
            color=[0.0, 100.0, 100.0],
            model=LightPositionModel.Global,
        )
    ]
    sim.set_light_setup(light_setup_2, "my_custom_lighting")

    # [/example 5]

    remove_all_objects(sim)

    # [example 6]

    id_1 = sim.add_object(chair_template_id, light_setup_key="my_custom_lighting")
    sim.set_rotation(mn.Quaternion.rotation(mn.Deg(-115), mn.Vector3.y_axis()), id_1)
    sim.set_translation([3.06, 0.47, 1.15], id_1)

    id_2 = sim.add_object(chair_template_id, light_setup_key="my_custom_lighting")
    sim.set_rotation(mn.Quaternion.rotation(mn.Deg(50), mn.Vector3.y_axis()), id_2)
    sim.set_translation([3.45927, 0.47, -0.624958], id_2)

    get_obs(sim, show_imgs, save_imgs)

    # [/example 6]

    # [example 7]
    existing_light_setup = sim.get_light_setup("my_custom_lighting")

    # create a new setup with an additional light
    new_light_setup = existing_light_setup + [
        LightInfo(
            vector=[0.0, 0.0, 1.0, 0.0],
            color=[1.6, 1.6, 1.4],
            model=LightPositionModel.Camera,
        )
    ]

    # register the old setup under a new name
    sim.set_light_setup(existing_light_setup, "my_old_custom_lighting")

    # [/example 7]

    # [example 8]

    # register the new setup overwriting the old one
    sim.set_light_setup(new_light_setup, "my_custom_lighting")

    get_obs(sim, show_imgs, save_imgs)

    # [/example 8]

    # [example 9]
    sim.set_object_light_setup(id_1, habitat_sim.gfx.DEFAULT_LIGHTING_KEY)

    get_obs(sim, show_imgs, save_imgs)

    # [/example 9]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-images", dest="show_images", action="store_false")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(show_images=True, save_images=True)
    args = parser.parse_args()
    main(show_imgs=args.show_images, save_imgs=args.save_images)
