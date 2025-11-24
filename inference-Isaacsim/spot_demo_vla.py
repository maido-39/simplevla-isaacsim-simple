# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Spot robot demo with VLA (Vision-Language-Action) model inference.
Uses trained VLA model to predict robot commands from camera images.
"""

# Launch Isaac Sim before any other imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# SimulationApp 초기화 후에만 다른 모듈을 import할 수 있습니다.
import sys
from pathlib import Path

# Add paths for imports
parent_dir = Path(__file__).parent.parent.parent
isaacsim_dir = parent_dir / "Isaacsim-spot-remotecontroldemo"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(isaacsim_dir))

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, Sdf
import omni
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from vla_controller import VLAController
import argparse


class SpotDemoVLA:
    """Spot robot demo with VLA model inference"""

    def __init__(self, checkpoint_path: str, instruction: str = "Navigate to the goal"):
        """
        Initialize the demo
        
        Args:
            checkpoint_path: Path to VLA model checkpoint
            instruction: Instruction text for the task
        """
        self.world = None
        self.stage = None
        self.spot = None
        self.controller = None
        self.physics_ready = False
        self.command_counter = 0
        self.checkpoint_path = checkpoint_path
        self.instruction = instruction
        
        # Camera paths
        self.robot_camera_path = None
        self.camera_path = None
        
        # Camera render products
        self.render_products = {}
        self.rgb_annotators = {}
        self.camera_render_initialized = False

    def initialize(self):
        """Initialize Isaac Sim world and stage"""
        # Create World: physics at 500Hz, rendering at 50Hz
        self.world = World(physics_dt=1.0/500.0, rendering_dt=10.0/500.0, stage_units_in_meters=1.0)
        # Get USD stage for scene manipulation
        self.stage = omni.usd.get_context().get_stage()
        print("World initialized")

    def setup_environment(self):
        """Setup basic environment: ground and sample box"""
        if self.world is None or self.stage is None:
            raise RuntimeError("World must be initialized first")

        # Add dome light for scene illumination
        omni.kit.commands.execute("CreatePrim", prim_path="/World/DomeLight", prim_type="DomeLight")
        light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
        if light_prim.IsValid():
            light_prim.GetAttribute("inputs:intensity").Set(600.0)

        # Create ground plane with physics properties
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/GroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        # Create sample box (dynamic, can be pushed by robot)
        box_position = np.array([2.0, 0.0, 0.25])
        box_scale = np.array([1.0, 1.0, 0.5])
        box_color = np.array([0.6, 0.4, 0.2])  # Brown color

        self.world.scene.add(DynamicCuboid(
            prim_path="/World/SampleBox",
            name="sample_box",
            position=box_position,
            scale=box_scale,
            color=box_color,
            mass=5.0,
            linear_velocity=np.array([0.0, 0.0, 0.0])
        ))

        # Apply physics material to box (friction and restitution)
        box_prim = self.stage.GetPrimAtPath("/World/SampleBox")
        if box_prim.IsValid():
            physics_material_path = "/World/Materials/BoxPhysicsMaterial"
            # Create physics material with friction properties
            physics_material = UsdPhysics.MaterialAPI.Apply(
                self.stage.DefinePrim(physics_material_path, "Material")
            )
            physics_material.CreateStaticFrictionAttr().Set(0.8)
            physics_material.CreateDynamicFrictionAttr().Set(0.7)
            physics_material.CreateRestitutionAttr().Set(0.1)

            # Bind physics material to box collider
            collider = UsdPhysics.CollisionAPI.Get(self.stage, "/World/SampleBox")
            if collider:
                collider.GetPrim().CreateRelationship("physics:material").SetTargets(
                    [Sdf.Path(physics_material_path)]
                )

        print("Environment setup complete: ground and sample box created")

    def setup_robot(self):
        """Setup Spot robot at origin"""
        # Add Spot robot to scene at origin
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion (no rotation)
        )
        print("Spot robot placed at origin")

    def _setup_robot_camera(self):
        """Create ego-view camera attached to Spot robot body"""
        try:
            # Check if robot body exists
            body_path = "/World/Spot/body"
            body_prim = self.stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                print(f"Warning: Robot body prim not found at {body_path}")
                return
            
            # Create camera using USD API
            camera_path = f"{body_path}/EgoCamera"
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            
            # Set camera parameters (RealSense D455 RGB specs)
            camera.GetFocalLengthAttr().Set(18.0)  # mm
            camera.GetHorizontalApertureAttr().Set(36.0)  # mm (90° H-FOV)
            camera.GetVerticalApertureAttr().Set(28.8)  # mm (65° V-FOV)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
            
            # Position camera relative to robot body
            camera_xform = UsdGeom.Xformable(camera)
            camera_xform.ClearXformOpOrder()
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(0.3, 0.0, 0.2))  # Position relative to robot body
            
            # Rotation: RPY (90°, -90°, 0°) -> quaternion
            rotation_quat = Gf.Quatf(0.5, 0.5, -0.5, -0.5)
            rotate_op = camera_xform.AddOrientOp()
            rotate_op.Set(rotation_quat)
            
            self.robot_camera_path = camera_path
            print(f"✓ Ego camera created: {camera_path}")
            
        except Exception as e:
            print(f"Error setting up robot camera: {e}")
            import traceback
            traceback.print_exc()

    def _setup_top_camera(self):
        """Create top-down camera"""
        try:
            
            # Create top camera
            camera_path = "/World/TopCamera"
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            
            # Set camera parameters
            camera.GetFocalLengthAttr().Set(18.0)  # mm
            camera.GetHorizontalApertureAttr().Set(18.0)  # mm
            camera.GetVerticalApertureAttr().Set(18.0)  # mm
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 50.0))
            
            # Position camera above origin, looking down
            camera_xform = UsdGeom.Xformable(camera)
            camera_xform.ClearXformOpOrder()
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(0.0, 0.0, 10.0))  # 10m above origin
            
            # Rotation: -90° around X axis (look down)
            rotation_quat = Gf.Quatf(0.707, -0.707, 0.0, 0.0)  # -90° around X
            rotate_op = camera_xform.AddOrientOp()
            rotate_op.Set(rotation_quat)
            
            self.camera_path = camera_path
            print(f"✓ Top camera created: {camera_path}")
            
        except Exception as e:
            print(f"Error setting up top camera: {e}")
            import traceback
            traceback.print_exc()

    def _initialize_camera_render_products(self):
        """Initialize render products and annotators for cameras"""
        try:
            import omni.replicator.core as rep
            
            # Initialize render products dictionary and annotators
            self.render_products = {}
            self.rgb_annotators = {}
            
            # Setup ego camera render product
            if self.robot_camera_path:
                ego_res = (640, 480)  # RealSense D455 RGB resolution
                self.render_products["ego"] = rep.create.render_product(
                    self.robot_camera_path, 
                    ego_res
                )
                self.rgb_annotators["ego"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["ego"].attach([self.render_products["ego"]])
                print(f"✓ Ego camera render product initialized: {ego_res[0]}×{ego_res[1]}")
            
            # Setup top camera render product
            if self.camera_path:
                top_res = (800, 800)
                self.render_products["top"] = rep.create.render_product(
                    self.camera_path, 
                    top_res
                )
                self.rgb_annotators["top"] = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotators["top"].attach([self.render_products["top"]])
                print(f"✓ Top camera render product initialized: {top_res[0]}×{top_res[1]}")
            
            self.camera_render_initialized = True
            
        except ImportError as e:
            print(f"Warning: Replicator not available, camera capture disabled: {e}")
            self.camera_render_initialized = False
        except Exception as e:
            print(f"Warning: Failed to initialize camera render products: {e}")
            self.camera_render_initialized = False

    def get_ego_camera_image(self):
        """Get current frame from ego camera"""
        if not self.camera_render_initialized:
            return None
        
        if "ego" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["ego"].get_data()
            if rgb_data is None:
                return None
            
            # Convert to numpy array
            image_array = np.asarray(rgb_data)
            
            if image_array is None or image_array.size == 0:
                return None
            
            # Handle RGBA to RGB conversion if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array
                
        except Exception as e:
            return None

    def get_top_camera_image(self):
        """Get current frame from top camera"""
        if not self.camera_render_initialized:
            return None
        
        if "top" not in self.rgb_annotators:
            return None
        
        try:
            rgb_data = self.rgb_annotators["top"].get_data()
            if rgb_data is None:
                return None
            
            # Convert to numpy array
            image_array = np.asarray(rgb_data)
            
            if image_array is None or image_array.size == 0:
                return None
            
            # Handle RGBA to RGB conversion if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array
            
        except Exception as e:
            return None

    def _on_physics_step(self, step_size):
        """
        Physics step callback - called every physics timestep (500Hz).
        Handles command updates (50Hz) and robot control.
        """
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
            # Robot is initialized, apply forward control with current command
            self.spot.forward(step_size, self.controller.get_command())
        else:
            # First physics step: initialize robot
            self.physics_ready = True
            self.spot.initialize()  # Initialize robot policy
            self.spot.post_reset()  # Post-reset setup
            self.spot.robot.set_joints_default_state(self.spot.default_pos)  # Set default joint positions
            print("Spot initialized")

    def setup(self):
        """Complete simulation setup: environment, robot, and controller"""
        # Initialize world if not already done
        if self.world is None:
            self.initialize()

        # Setup environment (ground and box)
        self.setup_environment()

        # Setup robot at origin
        self.setup_robot()

        # Create VLA controller
        self.controller = VLAController(
            checkpoint_path=self.checkpoint_path,
            instruction=self.instruction,
            max_vx=2.0, max_vy=2.0, max_yaw=2.0,
            inference_freq=2.0,  # Run inference at 2 Hz
            update_dt=0.02,  # 50Hz update rate
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        )

        # Reset world (required before querying articulation properties)
        self.world.reset()

        # Setup cameras (after world reset, robot is fully initialized)
        self._setup_robot_camera()
        self._setup_top_camera()
        
        # Initialize camera render products
        self._initialize_camera_render_products()

        # Register physics callback for robot control
        self.world.add_physics_callback("physics_step", callback_fn=self._on_physics_step)

        print("Setup complete")
        print(f"VLA Controller initialized with instruction: '{self.instruction}'")

    def run(self):
        """Run main simulation loop"""
        if self.world is None or self.spot is None or self.controller is None:
            raise RuntimeError("Simulation must be setup first")

        print("Starting simulation...")
        print("VLA model will predict robot commands from camera images")
        print("Press ESC in Isaac Sim window to quit")

        # Main simulation loop
        while simulation_app.is_running():
            # Check if quit requested
            if self.controller.is_quit_requested():
                break
            # Step physics and rendering
            self.world.step(render=True)

    def cleanup(self):
        """Cleanup resources: stop controller and remove physics callback"""
        # Stop controller
        if self.controller:
            self.controller.stop()
            
            # Print performance stats
            stats = self.controller.get_performance_stats()
            if stats:
                print("\nVLA Controller Performance Stats:")
                print(f"  Inference count: {stats['inference_count']}")
                print(f"  Avg inference time: {stats['avg_inference_time_ms']:.2f} ms")
                print(f"  Total inference time: {stats['total_inference_time_s']:.2f} s")

        # Remove physics callback
        if self.world and self.world.physics_callback_exists("physics_step"):
            self.world.remove_physics_callback("physics_step")

        print("Cleanup complete")


# ===================== Main Entry Point =====================
def main():
    """Main entry point for the demo"""
    parser = argparse.ArgumentParser(description="Spot Robot Demo with VLA Inference")
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
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = SpotDemoVLA(
        checkpoint_path=args.checkpoint,
        instruction=args.instruction
    )

    # Setup simulation (environment, robot, controller)
    demo.setup()

    # Run simulation loop
    try:
        demo.run()
    finally:
        # Always cleanup resources
        demo.cleanup()
        simulation_app.close()
        print("[INFO]: Simulation closed")


if __name__ == "__main__":
    main()

