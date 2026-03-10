# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ObjectNav V3 is an autonomous robot navigation system that uses visual SLAM, semantic mapping, and object detection to navigate indoor environments. It operates in the Habitat simulator (Meta AI's embodied AI platform) and can build semantic maps of environments and navigate to specified objects.

## Common Commands

### Running the Explorer (Map Building)
```bash
python explore_v2.py --scene /path/to/scene.glb --output-dir ./output_v2 --model-size n --confidence 0.4
```
- Interactively explore an environment and build semantic maps
- Controls: WASD=Move, Click map=Navigate, R=Scan, S=Save, Q=Quit
- Model sizes: n/s/m/l/x (n=fastest, x=most accurate)

### Running the Navigator (Using Pre-built Maps)
```bash
python navigate_v2.py --scene /path/to/scene.glb --map-dir ./output_v2
python navigate_v2.py --scene /path/to/scene.glb --map-dir ./output_v2 --use-vo
```
- Navigate to objects using pre-built maps
- `--use-vo` enables dead reckoning instead of ground truth pose

### Testing SLAM System
```bash
python scripts/test_slam.py --scene /path/to/scene.glb --output ./slam_results
python scripts/test_slam.py --scene /path/to/scene.glb --auto  # Automatic exploration
```

### Testing Visual Odometry
```bash
python scripts/test_vo.py --scene /path/to/scene.glb
python scripts/test_localization.py --scene /path/to/scene.glb
```

## Architecture Overview

### Core Components

**Perception Pipeline:**
- `perception/object_detector.py` - YOLOv8 object detection with 3D position extraction
- `mapping/enhanced_semantic_mapper_v2.py` - Main semantic mapper combining occupancy, objects, and multi-floor support
- `mapping/fixed_occupancy_map.py` - Depth-based occupancy grid with dynamic resizing

**SLAM System (`slam/`):**
- `slam_system.py` - Main VisualSLAM class integrating VO, loop closure, and pose graph
- `visual_odometry.py` - ORB features + ICP for pose estimation
- `loop_closure.py` - NetVLAD-style place recognition
- `pose_graph.py` - Graph-based optimization for trajectory correction
- `lightweight_vo.py`, `dead_reckoning.py` - Alternative lightweight pose tracking

**Navigation (`navigation/`):**
- `navigator.py` - Main navigator using A* planning with pre-built maps
- `smart_navigator.py` - Enhanced navigator with better floor transitions
- `global_planner.py`, `local_planner.py` - Hierarchical planning

**Core Types (`core/`):**
- `types.py` - Pose, Keyframe, MapPoint, TrackingState definitions
- `transforms.py` - Camera intrinsics, coordinate transformations
- `frontier.py` - Frontier-based exploration

### Coordinate System (Habitat)

Habitat uses Y-up coordinate system:
- X: right
- Y: up
- Z: backward (agent faces -Z at yaw=0)

Top-down map projection: Map X = World X, Map Y = World Z

World-to-map conversion uses origin offset stored in `statistics.json`:
```python
map_x = int((world_x - origin_x) * 100 / resolution + map_size // 2)
map_y = int((world_z - origin_z) * 100 / resolution + map_size // 2)
```

### Map Data Format

Maps are saved per-floor in `output_dir/floor_N/`:
- `navigable_map.pt` - PyTorch tensor, obstacles with stairs cleared
- `occupancy_map.pt` - Raw obstacle map
- `explored_map.pt` - Explored area mask
- `semantic_map.pt` - Multi-channel semantic categories
- `objects.json` - Detected objects with 3D positions

Global files:
- `statistics.json` - Map origin, resolution, size
- `floor_info.json` - Floor heights and object counts
- `objects.json` - All objects across floors

### Key Configuration

**MapConfig** (`mapping/fixed_occupancy_map.py`):
- `resolution_cm`: 5 (5cm per pixel default)
- `map_size_cm`: 2400 (24m x 24m default)
- `vision_range_cm`: 500 (5m max depth)

**Navigation thresholds** (`navigation/navigator.py`):
- `TURN_THRESHOLD`: 15 degrees
- `OBSTACLE_CLOSE`: 0.35m (critical)
- `OBSTACLE_THRESHOLD`: 0.6m (normal)
- `GOAL_TOLERANCE`: 12 pixels

### Dependencies

Required:
- habitat-sim (Habitat simulator)
- ultralytics (YOLOv8)
- opencv-python
- numpy
- torch

YOLO models download automatically on first use.
