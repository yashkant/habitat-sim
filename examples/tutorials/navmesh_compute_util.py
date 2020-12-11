# [setup]
import os

import numpy as np

import habitat_sim

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../../data")
output_path = os.path.join(dir_path, "navmesh_compute_output/")

save_index = 0


def make_configuration(scene_id):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False

    # agent configuration
    sensor_cfg = habitat_sim.SensorSpec()
    sensor_cfg.resolution = [256, 256]
    # center the camera at the agent position
    sensor_cfg.position = np.zeros(3)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def main_compute_navmesh(scene_dir):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    assets = {}
    missing_assets = []

    # fill the scene_ids
    for entry in os.listdir(scene_dir):
        asset_folder = os.path.join(scene_dir, entry)
        if not os.path.isdir(asset_folder):
            continue
        asset_id = entry[0:5]
        found_one = False
        for asset_file in os.listdir(asset_folder):
            full_asset_filename = os.path.join(asset_folder, asset_file)
            if os.path.isfile(full_asset_filename) and asset_file.endswith(".glb"):
                assets[asset_id] = full_asset_filename
                found_one = True
        if not found_one:
            missing_assets.append(asset_id)

    print(assets)

    if not assets:
        print("Nothing to do, aborting.")
        return

    cfg = make_configuration("NONE")
    sim = habitat_sim.Simulator(cfg)

    navMeshSettings = habitat_sim.NavMeshSettings()
    navMeshSettings.agent_height = 1.3

    # sort the house filename alphabetically
    for asset_id in sorted(assets.keys()):
        sim.close()
        cfg = make_configuration(assets[asset_id])
        cfg.sim_cfg.requires_textures = False
        sim.reconfigure(cfg)

        # recompute the NavMesh
        if sim.pathfinder == None:
            sim.pathfinder = habitat_sim.PathFinder()
        sim.recompute_navmesh(sim.pathfinder, navMeshSettings)

        # save the NavMesh
        if not sim.pathfinder.is_loaded:
            print(str(asset_id) + " - Failed to recompute the navmesh!!!!")
        else:
            sim.pathfinder.save_nav_mesh(
                os.path.join(output_path, str(asset_id) + ".navmesh")
            )
    print("___________________________________________\n")
    print("Processing Complete.")
    print("Missing assets: " + str(missing_assets))


def main_analyze_navmesh(navmesh_dir):
    # save a NavMesh_stats.csv in the output directory

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    navmesh_assets = {}

    # fill the scene_ids
    for entry in os.listdir(navmesh_dir):
        navmesh_file = os.path.join(navmesh_dir, entry)
        if os.path.isfile(navmesh_file):
            if not navmesh_file.endswith(".navmesh"):
                continue
            # assuming ##### format
            asset_id = entry[0:5]
            navmesh_assets[asset_id] = navmesh_file

    cfg = make_configuration("NONE")
    sim = habitat_sim.Simulator(cfg)
    sim.pathfinder = habitat_sim.PathFinder()

    navmesh_stats = {}

    # iteratively load each navmesh and compute the stats
    for asset_id in sorted(navmesh_assets.keys()):
        sim.pathfinder.load_nav_mesh(navmesh_assets[asset_id])
        if not sim.pathfinder.is_loaded:
            print(str(asset_id) + " - failed to load navmesh!!!")
            continue
        # collect the stats
        num_islands = sim.pathfinder.num_islands
        island_areas = []
        for island_ix in range(num_islands):
            island_areas.append(sim.pathfinder.island_area(island_ix))
        island_areas.sort()
        print(island_areas)
        max_island = island_areas[-1]
        print("max_island = " + str(max_island))
        second_max_island = max_island
        if num_islands > 1:
            second_max_island = island_areas[-2]
        island_delta = max_island - second_max_island
        island_delta_ratio = island_delta / max_island
        if island_delta_ratio < 0.0000001:
            island_delta_ratio = 1
        min_island = island_areas[0]

        navmesh_stats[asset_id] = [
            island_delta,
            island_delta_ratio,
            max_island,
            num_islands,
            min_island,
        ]

    # convert to contiguous csv
    f = open(output_path + "navmesh_analysis.csv", "w")
    f.write(
        "model_number, island_delta, island_delta_ratio, largest_island, num_islands, smallest_island\n"
    )
    for i in range(1006):
        asset_id = str(i).zfill(5)
        f.write(str(asset_id) + ",")
        if asset_id in navmesh_stats:
            asset_info = navmesh_stats[asset_id]
            for item in asset_info:
                f.write(str(item) + ",")
        f.write(" \n")
    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", type=str)
    parser.set_defaults()
    args = parser.parse_args()
    # main_compute_navmesh(args.dir)
    main_analyze_navmesh(args.dir)
