import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import re
from typing import List, Tuple, Dict


class VLADataset(Dataset):
    """
    Dataset for VLA model training.
    Loads images (ego, top), robot positions, and instructions.
    """
    
    def __init__(
        self,
        data_root: str,
        history_length: int = 5,
        trajectory_length: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        task_type: str = None  # 'box', 'moving', 'gate', or None for all
    ):
        """
        Args:
            data_root: Root directory containing the Proto_Data folder
            history_length: Number of historical frames to use (default: 5)
            trajectory_length: Length of output trajectory (default: 16)
            image_size: Size to resize images to
            task_type: Filter by task type if specified
        """
        self.data_root = data_root
        self.history_length = history_length
        self.trajectory_length = trajectory_length
        self.image_size = image_size
        
        # Load instruction files
        self.instructions = self._load_instructions()
        
        # Find all task directories
        self.samples = self._load_samples(task_type)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_instructions(self) -> Dict[str, Dict[str, str]]:
        """Load instructions from all Instruction.txt files"""
        instructions = {}
        
        # Map person names to their instruction files
        person_files = {
            'Seongmin Hong': 'Seongmin Hong_Instruction.txt',
            'Byeonggyu Park': 'Byeonggyu Park_Instruction.txt',
            'Jihun Mun': 'Jihun Mun_Instruction.txt',
            'Jiyoon Gong': 'Jiyoon Gong_Instruction.txt'
        }
        
        for person, filename in person_files.items():
            filepath = os.path.join(self.data_root, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse instructions for each task
                task_instructions = {}
                for task in ['Moving', 'Box', 'Gate']:
                    pattern = f'Task: {task}\\nDescription: ([^\\n]+)\\n((?:[^\\n]+\\n?)+)'
                    match = re.search(pattern, content)
                    if match:
                        description = match.group(1)
                        steps = match.group(2).strip()
                        task_instructions[task.lower()] = f"{description}. {steps}"
                
                instructions[person] = task_instructions
        
        return instructions
    
    def _load_samples(self, task_type: str = None) -> List[Dict]:
        """Load all data samples from task directories"""
        samples = []
        
        # Get all directories in data_root
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            if not os.path.isdir(item_path):
                continue
            
            # Skip if it's not a task directory (format: YYMMDD_HHMMSS-person_task)
            if not re.match(r'\d{6}_\d{6}-[\w]+_(box|moving|gate)', item):
                continue
            
            # Extract task type from directory name
            match = re.search(r'_(box|moving|gate)$', item)
            if not match:
                continue
            
            task = match.group(1)
            if task_type and task != task_type:
                continue
            
            # Extract person name
            person_match = re.search(r'-\d*([A-Za-z]+)', item)
            person_name = None
            if person_match:
                name_part = person_match.group(1)
                # Map to full name
                name_mapping = {
                    'Hongseongmin': 'Seongmin Hong',
                    'ByeonggyuPark': 'Byeonggyu Park',
                    'moonjihun': 'Jihun Mun',
                    'jiyoon': 'Jiyoon Gong'
                }
                person_name = name_mapping.get(name_part, None)
            
            # Check if camera and data.csv exist
            camera_dir = os.path.join(item_path, 'camera')
            csv_path = os.path.join(item_path, 'data.csv')
            
            if not os.path.exists(camera_dir) or not os.path.exists(csv_path):
                continue
            
            # Load CSV data
            try:
                df = pd.read_csv(csv_path)
                if len(df) < self.history_length + self.trajectory_length:
                    continue
                
                # Get instruction
                instruction = None
                if person_name and person_name in self.instructions:
                    instruction = self.instructions[person_name].get(task, "")
                
                # Create samples with sliding window
                for i in range(len(df) - self.history_length - self.trajectory_length + 1):
                    samples.append({
                        'task_dir': item_path,
                        'task': task,
                        'person': person_name,
                        'instruction': instruction,
                        'start_idx': i,
                        'end_idx': i + self.history_length + self.trajectory_length
                    })
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue
        
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess an image"""
        if not os.path.exists(image_path):
            # Return black image if not found
            return Image.new('RGB', self.image_size, (0, 0, 0))
        
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', self.image_size, (0, 0, 0))
    
    def _get_image_path(self, task_dir: str, camera_type: str, frame_num: int) -> str:
        """Get image path for a specific frame"""
        camera_dir = os.path.join(task_dir, 'camera', camera_type)
        
        # Try to find the image file
        # Format: frame{frame_num}-{timestamp}-{camera_type}.jpg
        for filename in os.listdir(camera_dir):
            if filename.endswith('.jpg'):
                # Extract frame number from filename
                match = re.search(r'frame(\d+)', filename)
                if match and int(match.group(1)) == frame_num:
                    return os.path.join(camera_dir, filename)
        
        # Fallback: try direct frame number
        potential_path = os.path.join(camera_dir, f'frame{frame_num}-{camera_type}.jpg')
        if os.path.exists(potential_path):
            return potential_path
        
        # Return None if not found
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        task_dir = sample['task_dir']
        start_idx = sample['start_idx']
        
        # Load CSV data
        csv_path = os.path.join(task_dir, 'data.csv')
        df = pd.read_csv(csv_path)
        
        # Extract history indices and trajectory indices
        history_indices = list(range(start_idx, start_idx + self.history_length))
        trajectory_start = start_idx + self.history_length
        trajectory_indices = list(range(trajectory_start, trajectory_start + self.trajectory_length))
        
        # Load images (history)
        ego_images = []
        top_images = []
        
        for i in history_indices:
            if i < len(df):
                frame_num = int(df.iloc[i]['frame_num'])
                
                # Load ego image
                ego_path = self._get_image_path(task_dir, 'ego', frame_num)
                ego_img = self._load_image(ego_path)
                ego_images.append(ego_img)
                
                # Load top image
                top_path = self._get_image_path(task_dir, 'top', frame_num)
                top_img = self._load_image(top_path)
                top_images.append(top_img)
            else:
                # Pad with black images if needed
                ego_images.append(Image.new('RGB', self.image_size, (0, 0, 0)))
                top_images.append(Image.new('RGB', self.image_size, (0, 0, 0)))
        
        # Load robot positions for trajectory
        robot_positions = []
        robot_orientations = []
        timestamps = []
        dt_values = []  # Delta time (time difference between consecutive frames)
        
        prev_timestamp = None
        for i in trajectory_indices:
            if i < len(df):
                pos_x = float(df.iloc[i]['robot_pos_x'])
                pos_y = float(df.iloc[i]['robot_pos_y'])
                pos_z = float(df.iloc[i]['robot_pos_z'])
                robot_positions.append([pos_x, pos_y, pos_z])
                
                orient_w = float(df.iloc[i]['robot_orient_w'])
                orient_x = float(df.iloc[i]['robot_orient_x'])
                orient_y = float(df.iloc[i]['robot_orient_y'])
                orient_z = float(df.iloc[i]['robot_orient_z'])
                robot_orientations.append([orient_w, orient_x, orient_y, orient_z])
                
                # Extract timestamp
                timestamp = float(df.iloc[i]['timestamp'])
                timestamps.append(timestamp)
                
                # Calculate dt (delta time from previous frame)
                if prev_timestamp is not None:
                    dt = timestamp - prev_timestamp
                    dt_values.append(dt)
                else:
                    # First frame: dt is 0 or use first dt from history if available
                    if start_idx > 0 and start_idx - 1 < len(df):
                        prev_hist_timestamp = float(df.iloc[start_idx - 1]['timestamp'])
                        dt = timestamp - prev_hist_timestamp
                    else:
                        dt = 0.0
                    dt_values.append(dt)
                
                prev_timestamp = timestamp
            else:
                # Pad with last known position
                if robot_positions and timestamps:
                    robot_positions.append(robot_positions[-1])
                    robot_orientations.append(robot_orientations[-1])
                    timestamps.append(timestamps[-1])
                    # Use last dt value or 0.0 if not available
                    dt_values.append(dt_values[-1] if dt_values else 0.0)
                else:
                    robot_positions.append([0.0, 0.0, 0.0])
                    robot_orientations.append([1.0, 0.0, 0.0, 0.0])
                    timestamps.append(0.0)
                    dt_values.append(0.0)
        
        # Also extract timestamps for history (for temporal context)
        history_timestamps = []
        for i in history_indices:
            if i < len(df):
                timestamp = float(df.iloc[i]['timestamp'])
                history_timestamps.append(timestamp)
            else:
                history_timestamps.append(0.0)
        
        # Get instruction
        instruction = sample.get('instruction', "")
        if not instruction:
            instruction = f"Perform {sample['task']} task."
        
        return {
            'ego_images': ego_images,
            'top_images': top_images,
            'robot_positions': np.array(robot_positions, dtype=np.float32),
            'robot_orientations': np.array(robot_orientations, dtype=np.float32),
            'timestamps': np.array(timestamps, dtype=np.float32),
            'dt': np.array(dt_values, dtype=np.float32),
            'history_timestamps': np.array(history_timestamps, dtype=np.float32),
            'instruction': instruction,
            'task': sample['task']
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    # Stack images
    ego_images_list = [torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 
                                    for img in item['ego_images']]) for item in batch]
    top_images_list = [torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 
                                    for img in item['top_images']]) for item in batch]
    
    # Pad sequences to same length
    max_history = max([img.shape[0] for img in ego_images_list])
    padded_ego = []
    padded_top = []
    
    for ego, top in zip(ego_images_list, top_images_list):
        if ego.shape[0] < max_history:
            pad_size = max_history - ego.shape[0]
            pad = torch.zeros(pad_size, *ego.shape[1:])
            ego = torch.cat([ego, pad], dim=0)
            top = torch.cat([top, pad], dim=0)
        padded_ego.append(ego)
        padded_top.append(top)
    
    ego_images = torch.stack(padded_ego)
    top_images = torch.stack(padded_top)
    
    # Stack trajectories
    robot_positions = torch.from_numpy(np.stack([item['robot_positions'] for item in batch]))
    robot_orientations = torch.from_numpy(np.stack([item['robot_orientations'] for item in batch]))
    
    # Stack timestamps and dt
    timestamps = torch.from_numpy(np.stack([item['timestamps'] for item in batch]))
    dt = torch.from_numpy(np.stack([item['dt'] for item in batch]))
    history_timestamps = torch.from_numpy(np.stack([item['history_timestamps'] for item in batch]))
    
    # Instructions
    instructions = [item['instruction'] for item in batch]
    tasks = [item['task'] for item in batch]
    
    return {
        'ego_images': ego_images,
        'top_images': top_images,
        'robot_positions': robot_positions,
        'robot_orientations': robot_orientations,
        'timestamps': timestamps,
        'dt': dt,
        'history_timestamps': history_timestamps,
        'instructions': instructions,
        'tasks': tasks
    }

