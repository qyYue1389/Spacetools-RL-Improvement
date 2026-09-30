import ray, time, sys
ray.init(address="auto")
from toolshed import start_toolkit
roborefer_model, depth_checkpoint = sys.argv[1], sys.argv[2]
TOOL_CONFIGS = {
    'roborefer': {'num_actors': 1, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
        'args': {'model_path': roborefer_model, 'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'auto', 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'depth_estimator': {'num_actors': 2, 'resources': {'num_gpus': 0.4}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'checkpoint_path': depth_checkpoint, 'no_output_image': True, 'no_output_vars': False}},
    'bounding_box': {'num_actors': 2, 'resources': {'num_gpus': 0.05}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'vision_ops': {'num_actors': 1, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['mask_crop', 'point_crop', 'project_2d_points_to_3d', 'project_3d_points_to_2d', 'draw_polygon'], 'exclude_behavior': 'error'}},
    'grasp_generator': {'num_actors': 1, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-graspgen', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
}
router = start_toolkit(TOOL_CONFIGS, detached=False, dashboard=False)
while True: time.sleep(60)
