# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Isaac Sim Spot robot simulation with VLA (Vision-Language-Action) model inference.
Full environment with randomization, gates, boxes, and camera visualization.
Uses trained VLA model to predict robot commands from camera images.
"""

# Launch Isaac Sim before any other imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})  # Headless mode for Pygame display

# SimulationApp 초기화 후에만 다른 모듈을 import할 수 있습니다.
import sys
from pathlib import Path

# Add paths for imports
parent_dir = Path(__file__).parent.parent.parent
isaacsim_dir = parent_dir / "Isaacsim-spot-remotecontroldemo"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(isaacsim_dir))

import numpy as np
import logging
import time
import queue
import threading
import pygame
import json
import os
import csv
import argparse
from datetime import datetime
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Sdf
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PreviewSurface
import omni.kit.commands
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from vla_controller import VLAController
import colorsys

# Import PygameDualCameraDisplay from quadruped_example (reference only, we'll recreate it)
# We'll copy the display class here to avoid editing the original file


# ===================== Pygame Display for Dual Camera =====================
class PygameDualCameraDisplay:
    """
    High-performance Pygame display for dual camera streaming (ego + top view).
    Uses separate thread to avoid blocking main simulation loop.
    """
    
    def __init__(self, window_size=(1600, 800), window_title="Spot Robot - Ego & Top View", 
                 ego_camera_resolution=(640, 480)):
        """
        Initialize dual camera display.
        
        Args:
            window_size: Total window size (width, height) - will be split in half
            window_title: Window title
            ego_camera_resolution: Original ego camera resolution (width, height) for aspect ratio calculation
        """
        self.window_size = window_size
        self.window_title = window_title
        self.ego_frame_queue = queue.Queue(maxsize=2)  # Keep only latest 2 frames
        self.top_frame_queue = queue.Queue(maxsize=2)  # Keep only latest 2 frames
        self.running = False
        self.display_thread = None
        self.screen = None
        self.clock = None
        self.ego_camera_resolution = ego_camera_resolution
        
        # Calculate individual camera display sizes (side by side)
        self.camera_width = window_size[0] // 2
        self.camera_height = window_size[1]
        
    def start(self):
        """Start display thread"""
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        time.sleep(0.1)  # Give thread time to initialize
        print(f"✓ Pygame dual camera display started: {self.window_size[0]}×{self.window_size[1]}")
    
    def stop(self):
        """Stop display thread"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        if self.screen:
            pygame.quit()
        print("Pygame display stopped")
    
    def update_ego_frame(self, image_array):
        """Update ego camera frame (non-blocking, drops old frames if queue full)"""
        if not self.running:
            return
        self._update_frame_queue(
            self.ego_frame_queue, 
            image_array, 
            (self.camera_width, self.camera_height),
            maintain_aspect=True,
            scale_to_fill=True
        )
    
    def update_top_frame(self, image_array):
        """Update top camera frame (non-blocking, drops old frames if queue full)"""
        if not self.running:
            return
        self._update_frame_queue(
            self.top_frame_queue, 
            image_array, 
            (self.camera_width, self.camera_height),
            maintain_aspect=True,
            scale_to_fill=True
        )
    
    def _update_frame_queue(self, frame_queue, image_array, target_size, maintain_aspect=False, scale_to_fill=True):
        """Helper method to update a frame queue"""
        try:
            # Ensure image is in correct format: (H, W, 3) RGB uint8
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
            
            # Convert to pygame surface
            image_swapped = np.swapaxes(image_array, 0, 1)
            surface = pygame.surfarray.make_surface(image_swapped)
            
            # Resize if needed
            if surface.get_size() != target_size:
                if maintain_aspect and scale_to_fill:
                    img_w, img_h = surface.get_size()
                    target_w, target_h = target_size
                    
                    scale_w = target_w / img_w
                    scale_h = target_h / img_h
                    scale = max(scale_w, scale_h)
                    
                    scaled_w = int(img_w * scale)
                    scaled_h = int(img_h * scale)
                    surface = pygame.transform.smoothscale(surface, (scaled_w, scaled_h))
                    
                    if scaled_w != target_w or scaled_h != target_h:
                        crop_x = (scaled_w - target_w) // 2
                        crop_y = (scaled_h - target_h) // 2
                        surface = surface.subsurface((crop_x, crop_y, target_w, target_h))
                elif maintain_aspect:
                    surface = pygame.transform.smoothscale(surface, target_size)
                else:
                    surface = pygame.transform.scale(surface, target_size)
            
            # Put in queue (non-blocking, drop if full)
            try:
                frame_queue.put_nowait(surface)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(surface)
                except queue.Empty:
                    pass
        except Exception as e:
            pass
    
    def _display_loop(self):
        """Display loop running in separate thread"""
        try:
            pygame.init()
            flags = pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.HWSURFACE
            self.screen = pygame.display.set_mode(self.window_size, flags)
            pygame.display.set_caption(self.window_title)
            self.clock = pygame.time.Clock()
            
            target_fps = 60
            font = pygame.font.Font(None, 36)
            ego_label = font.render("Ego View (VLA)", True, (255, 255, 255))
            top_label = font.render("Top View", True, (255, 255, 255))
            
            back_buffer = pygame.Surface(self.window_size)
            last_ego_surface = None
            last_top_surface = None
            
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                            break
                    elif event.type == pygame.VIDEORESIZE:
                        self.window_size = event.size
                        self.camera_width = self.window_size[0] // 2
                        self.camera_height = self.window_size[1]
                        flags = pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.HWSURFACE
                        self.screen = pygame.display.set_mode(self.window_size, flags)
                        back_buffer = pygame.Surface(self.window_size)
                
                # Get latest frames from queues
                ego_surface = None
                top_surface = None
                
                try:
                    while True:
                        ego_surface = self.ego_frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                try:
                    while True:
                        top_surface = self.top_frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                if ego_surface is None:
                    ego_surface = last_ego_surface
                else:
                    last_ego_surface = ego_surface
                
                if top_surface is None:
                    top_surface = last_top_surface
                else:
                    last_top_surface = top_surface
                
                # Draw to back buffer
                back_buffer.fill((0, 0, 0))
                
                if ego_surface:
                    back_buffer.blit(ego_surface, (0, 0))
                else:
                    pygame.draw.rect(back_buffer, (50, 50, 50), (0, 0, self.camera_width, self.camera_height))
                
                if top_surface:
                    back_buffer.blit(top_surface, (self.camera_width, 0))
                else:
                    pygame.draw.rect(back_buffer, (50, 50, 50), (self.camera_width, 0, self.camera_width, self.camera_height))
                
                # Draw labels
                back_buffer.blit(ego_label, (10, 10))
                back_buffer.blit(top_label, (self.camera_width + 10, 10))
                
                # Draw divider line
                pygame.draw.line(back_buffer, (100, 100, 100), 
                               (self.camera_width, 0), 
                               (self.camera_width, self.camera_height), 2)
                
                # Blit back buffer to screen
                self.screen.blit(back_buffer, (0, 0))
                pygame.display.flip()
                
                self.clock.tick(target_fps)
                
        except Exception as e:
            print(f"Pygame display error: {e}")
        finally:
            if self.screen:
                pygame.quit()


# ===================== Default Configuration =====================
# Copy default config from quadruped_example.py (reference only)
DEFAULT_CONFIG = {
    "ground_color": [0.5, 0.5, 0.5],
    "wall_color": [0.7, 0.7, 0.7],
    "wall_height": 2.0,
    "map_size": 10.0,
    "ground_friction_static": 0.2,
    "ground_friction_dynamic": 0.2,
    "ground_restitution": 0.01,
    "box_friction_static": 0.8,
    "box_friction_dynamic": 0.7,
    "box_restitution": 0.1,
    "dome_light_intensity": 600.0,
    "marker_radius": 0.3,
    "goal_hemisphere_diameter": 1.0,
    "object_type": "box",
    "wall_inset": 1.0,
    "box_line_distance_min": 2.0,
    "box_line_distance_max": 3.0,
    "box_scale_range": [[0.8, 2.0], [0.8, 2.0], [0.5, 1.0]],
    "box_mass_range": [3.0, 10.0],
    "num_boxes": 2,
    "box_color_range": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    "box_min_separation": 1.5,
    "max_vx": 2.0,
    "max_vy": 2.0,
    "max_yaw": 2.0,
    "acc_vx": 5.0,
    "acc_vy": 5.0,
    "acc_yaw": 10.0,
    "decay_vx": 0.7,
    "decay_vy": 0.7,
    "decay_yaw": 0.6,
    "update_dt": 0.02,
    "ego_camera_resolution": [640, 480],
    "top_camera_resolution": [800, 800],
    "top_camera_height": 10.0,
    "randomize": True,
    "random_seed": None,
    "start_position": [4.0, 4.0],
    "goal_position": [-4.0, -4.0],
    "box_position": [2.0, 0.0, 0.25],
    "box_scale": [1.0, 1.0, 0.5],
    "box_mass": 5.0,
    "box_color": [0.6, 0.4, 0.2],
    "robot_height": 0.8,
}


# ===================== Main Simulation Class =====================
class SpotSimulationVLA:
    """Main simulation class for Isaac Sim Spot robot control with VLA inference"""

    def __init__(self, checkpoint_path: str, instruction: str = "Navigate to the goal",
                 config_file=None, experiment_name=None, log_level=logging.INFO, 
                 enable_csv_logging=True, enable_image_saving=True, **config_overrides):
        """
        Initialize simulation.
    
        Args:
            checkpoint_path: Path to VLA model checkpoint
            instruction: Instruction text for the task
            config_file: Path to JSON config file (optional)
            experiment_name: Name of the experiment (optional, defaults to "NULL")
            log_level: Logging level (default: logging.INFO)
            enable_csv_logging: Enable CSV data logging (default: True)
            enable_image_saving: Enable camera image saving (default: True)
            **config_overrides: Override any config values
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        if config_file:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        self.config.update(config_overrides)
        
        # Store VLA parameters
        self.checkpoint_path = checkpoint_path
        self.instruction = instruction
        
        # Setup logging system
        self._setup_logging(log_level)
        
        # Initialize Isaac Sim components
        self.world = None
        self.stage = None
        self.spot = None
        self.controller = None
        
        # Simulation state variables
        self.physics_ready = False
        self.command_counter = 0
        self.logging_counter = 0
        self.start_pos = None
        self.goal_pos = None
        self.robot_camera_path = None
        self.camera_path = None
        self.display = None
        
        # Experiment data tracking
        self.experiment_name = experiment_name if experiment_name else "NULL"
        self.experiment_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_counter = 0
        self.experiment_start_time = None
        self.data_saving_started = False
        self.first_command_received = False
        
        self.enable_csv_logging = enable_csv_logging
        self.enable_image_saving = enable_image_saving
        
        # Camera render products
        self.render_products = {}
        self.rgb_annotators = {}
        self.camera_render_initialized = False
        
        # Random number generator
        self._rng = None

    def _setup_logging(self, log_level=logging.INFO):
        """Setup logging system with console output"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    # Note: We'll need to copy many methods from quadruped_example.py
    # For brevity, I'll include the essential ones and reference the pattern
    # The full implementation would include all environment setup, randomization, etc.
    
    def initialize(self):
        """Initialize Isaac Sim world and stage"""
        self.world = World(physics_dt=1.0/500.0, rendering_dt=10.0/500.0, stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        self.logger.info("World initialized")

    def setup(self):
        """Complete simulation setup"""
        if self.world is None:
            self.initialize()
        
        # Setup environment (would call setup_environment from quadruped_example pattern)
        # For now, we'll create a simplified version
        self._setup_basic_environment()
        
        # Setup robot
        self._setup_basic_robot()
        
        # Create VLA controller
        cfg = self.config
        self.controller = VLAController(
            checkpoint_path=self.checkpoint_path,
            instruction=self.instruction,
            max_vx=cfg["max_vx"], max_vy=cfg["max_vy"], max_yaw=cfg["max_yaw"],
            inference_freq=2.0,  # Run inference at 2 Hz
            update_dt=cfg["update_dt"],
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        )
        
        # Reset world
        self.world.reset()
        
        # Setup cameras
        self._setup_robot_camera()
        self._setup_top_camera()
        self._initialize_camera_render_products()
        
        # Initialize Pygame display
        window_width = 1600
        window_height = 800
        ego_res = self.config["ego_camera_resolution"]
        
        self.display = PygameDualCameraDisplay(
            window_size=(window_width, window_height),
            window_title=f"Spot Robot - VLA Inference: '{self.instruction}'",
            ego_camera_resolution=ego_res
        )
        self.display.start()
        
        # Register physics callback
        self.world.add_physics_callback("physics_step", callback_fn=self._on_physics_step)
        
        self.logger.info("Setup complete")
        self.logger.info(f"VLA Controller initialized with instruction: '{self.instruction}'")

    def _setup_basic_environment(self):
        """Setup basic environment (simplified version)"""
        # Add dome light
        omni.kit.commands.execute("CreatePrim", prim_path="/World/DomeLight", prim_type="DomeLight")
        light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
        if light_prim.IsValid():
            light_prim.GetAttribute("inputs:intensity").Set(self.config["dome_light_intensity"])
        
        # Create ground plane
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/GroundPlane",
            static_friction=self.config["ground_friction_static"],
            dynamic_friction=self.config["ground_friction_dynamic"],
            restitution=self.config["ground_restitution"],
        )
        
        # Store positions for robot setup
        self.start_pos = np.array(self.config["start_position"])
        self.goal_pos = np.array(self.config["goal_position"])
        
        self.logger.info("Basic environment setup complete")

    def _setup_basic_robot(self):
        """Setup robot at start position"""
        if self.start_pos is None or self.goal_pos is None:
            self.start_pos = np.array([0.0, 0.0])
            self.goal_pos = np.array([-4.0, -4.0])
        
        direction = self.goal_pos - self.start_pos
        yaw = np.arctan2(direction[1], direction[0])
        half_yaw = yaw / 2.0
        orientation = np.array([
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)
        ])
        
        robot_height = self.config.get("robot_height", 0.65)
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([self.start_pos[0], self.start_pos[1], robot_height]),
            orientation=orientation,
        )
        
        self.logger.info(f"Robot placed at [{self.start_pos[0]:.2f}, {self.start_pos[1]:.2f}]")

    def _setup_robot_camera(self):
        """Create ego-view camera attached to Spot robot body"""
        try:
            body_path = "/World/Spot/body"
            body_prim = self.stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                self.logger.warning(f"Robot body prim not found at {body_path}")
                return
            
            camera_path = f"{body_path}/EgoCamera"
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            
            camera.GetFocalLengthAttr().Set(18.0)
            camera.GetHorizontalApertureAttr().Set(36.0)
            camera.GetVerticalApertureAttr().Set(28.8)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
            
            camera_xform = UsdGeom.Xformable(camera)
            camera_xform.ClearXformOpOrder()
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(0.3, 0.0, 0.2))
            
            rotation_quat = Gf.Quatf(0.5, 0.5, -0.5, -0.5)
            rotate_op = camera_xform.AddOrientOp()
            rotate_op.Set(rotation_quat)
            
            self.robot_camera_path = camera_path
            self.logger.info(f"✓ Ego camera created: {camera_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up robot camera: {e}")

    def _setup_top_camera(self):
        """Create top-down camera"""
        try:
            camera_path = "/World/TopCamera"
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            
            camera.GetFocalLengthAttr().Set(18.0)
            camera.GetHorizontalApertureAttr().Set(18.0)
            camera.GetVerticalApertureAttr().Set(18.0)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 50.0))
            
            camera_xform = UsdGeom.Xformable(camera)
            camera_xform.ClearXformOpOrder()
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(0.0, 0.0, self.config["top_camera_height"]))
            
            rotation_quat = Gf.Quatf(0.707, -0.707, 0.0, 0.0)
            rotate_op = camera_xform.AddOrientOp()
            rotate_op.Set(rotation_quat)
            
            self.camera_path = camera_path
            self.logger.info(f"✓ Top camera created: {camera_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up top camera: {e}")

    def _initialize_camera_render_products(self):
        """Initialize render products and annotators for cameras"""
        try:
            import omni.replicator.core as rep
            
            self.render_products = {}
            self.rgb_annotators = {}
            
            if self.robot_camera_path:
                ego_res = tuple(self.config["ego_camera_resolution"])
                self.render_products["ego"] = rep.create.render_product(
                    self.robot_camera_path, 
                    ego_res
                )
                self.rgb_annotators["ego"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["ego"].attach([self.render_products["ego"]])
                self.logger.info(f"✓ Ego camera render product initialized: {ego_res[0]}×{ego_res[1]}")
            
            if self.camera_path:
                top_res = tuple(self.config["top_camera_resolution"])
                self.render_products["top"] = rep.create.render_product(
                    self.camera_path, 
                    top_res
                )
                self.rgb_annotators["top"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["top"].attach([self.render_products["top"]])
                self.logger.info(f"✓ Top camera render product initialized: {top_res[0]}×{top_res[1]}")
            
            self.camera_render_initialized = True
            
        except ImportError as e:
            self.logger.warning(f"Replicator not available: {e}")
            self.camera_render_initialized = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize camera render products: {e}")
            self.camera_render_initialized = False

    def get_ego_camera_image(self):
        """Get current frame from ego camera"""
        if not self.camera_render_initialized or "ego" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["ego"].get_data()
            if rgb_data is None:
                return None
            
            image_array = np.asarray(rgb_data)
            if image_array is None or image_array.size == 0:
                return None
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array
        except Exception as e:
            return None

    def get_top_camera_image(self):
        """Get current frame from top camera"""
        if not self.camera_render_initialized or "top" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["top"].get_data()
            if rgb_data is None:
                return None
            
            image_array = np.asarray(rgb_data)
            if image_array is None or image_array.size == 0:
                return None
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array
        except Exception as e:
            return None

    def _on_physics_step(self, step_size):
        """Physics step callback - called every physics timestep (500Hz)"""
        # Command update: update controller at 50Hz (every 10 physics steps)
        self.command_counter += 1
        if self.command_counter >= 10:
            self.command_counter = 0
            
            # Update image history with latest camera frames
            ego_image = self.get_ego_camera_image()
            top_image = self.get_top_camera_image()
            
            if ego_image is not None and top_image is not None:
                self.controller.update_image_history(ego_image, top_image)
            
            # Update controller (runs inference at specified frequency)
            self.controller.update()

        # Robot control: apply commands to robot
        if self.physics_ready:
            self.spot.forward(step_size, self.controller.get_command())
        else:
            self.physics_ready = True
            self.spot.initialize()
            self.spot.post_reset()
            self.spot.robot.set_joints_default_state(self.spot.default_pos)
            self.logger.info("Spot initialized")

    def run(self):
        """Run main simulation loop"""
        if self.world is None or self.spot is None or self.controller is None:
            raise RuntimeError("Simulation must be setup first")
        
        self.logger.info("Starting simulation...")
        self.logger.info("VLA model will predict robot commands from camera images")
        self.logger.info("Press ESC in Pygame window to quit")
        
        while simulation_app.is_running():
            if self.controller.is_quit_requested():
                break
            
            if self.display and not self.display.running:
                self.logger.info("Pygame window closed - quitting...")
                break
            
            # Step physics and rendering
            self.world.step(render=True)
            
            # Update Pygame display with camera frames
            if self.display:
                ego_image = self.get_ego_camera_image()
                if ego_image is not None:
                    self.display.update_ego_frame(ego_image)
                
                top_image = self.get_top_camera_image()
                if top_image is not None:
                    self.display.update_top_frame(top_image)

    def cleanup(self):
        """Cleanup resources"""
        if self.display:
            self.display.stop()
        
        if self.controller:
            self.controller.stop()
            
            # Print performance stats
            stats = self.controller.get_performance_stats()
            if stats:
                self.logger.info("\nVLA Controller Performance Stats:")
                self.logger.info(f"  Inference count: {stats['inference_count']}")
                self.logger.info(f"  Avg inference time: {stats['avg_inference_time_ms']:.2f} ms")
                self.logger.info(f"  Total inference time: {stats['total_inference_time_s']:.2f} s")
        
        if self.world and self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")
        
        self.logger.info("Cleanup complete")


# ===================== Main Entry Point =====================
def main():
    """Main entry point for the simulation"""
    parser = argparse.ArgumentParser(description="Isaac Sim - Spot Robot VLA Inference Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VLA model checkpoint"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Navigate to the goal",
        help="Instruction text for the task"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="VLA_TEST",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map[args.loglevel.upper()]
    
    # Create simulation instance
    sim = SpotSimulationVLA(
        checkpoint_path=args.checkpoint,
        instruction=args.instruction,
        config_file=args.config,
        experiment_name=args.experiment_name,
        log_level=log_level
    )
    
    # Setup simulation
    sim.setup()
    
    # Run simulation loop
    try:
        sim.run()
    finally:
        sim.cleanup()
        simulation_app.close()
        print("[INFO]: Simulation closed")


if __name__ == "__main__":
    main()

