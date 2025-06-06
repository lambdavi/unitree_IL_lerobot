import time
import numpy as np
from datasets import load_dataset
from multiprocessing import shared_memory, Array, Lock
import threading

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller
from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient

class DatasetReplayer:
    def __init__(self):
        # Initialize the dataset
        print("Loading pouring dataset...")
        self.dataset = load_dataset("unitreerobotics/G1_Pouring_Dataset")
        self.train_dataset = self.dataset["train"]
        
        # Get the features to understand the data structure
        self.features = self.train_dataset.features
        print("Dataset features:", self.features)
        
        # Define joint names for reference
        self.joint_names = [
            "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
            "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
            "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
            "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
            "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
            "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
            "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
            "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"
        ]
        
        # Initialize robot controllers
        self.arm_ctrl = G1_29_ArmController()
        
        # Initialize hand controllers
        self.left_hand_array = Array('d', 7, lock=True)
        self.right_hand_array = Array('d', 7, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array('d', 14, lock=False)
        self.dual_hand_action_array = Array('d', 14, lock=False)
        self.hand_ctrl = Dex3_1_Controller(self.left_hand_array, self.right_hand_array, 
                                         self.dual_hand_data_lock, self.dual_hand_state_array, 
                                         self.dual_hand_action_array)

        # Initialize image client
        self._setup_image_client()
        
        # Initialize timing variables
        self.last_step_time = None
        self.step_count = 0
        self.total_execution_time = 0

    def _setup_image_client(self):
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 1280],
            'head_camera_id_numbers': [0],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],
            'wrist_camera_id_numbers': [2, 4],
        }

        # Setup image client
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        
        self.tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
        self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_img_shm.buf)
        
        self.wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_img_shm.buf)
        
        self.img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=self.tv_img_shm.name,
                                    wrist_img_shape=wrist_img_shape, wrist_img_shm_name=self.wrist_img_shm.name)

        # Start image receiving thread
        self.image_receive_thread = threading.Thread(target=self.img_client.receive_process, daemon=True)
        self.image_receive_thread.start()

    def get_current_state(self, episode):
        """Get the current state from the episode"""
        state = episode['observation.state']
        action = episode['action']
        
        # Create a dictionary mapping joint names to their values
        state_dict = dict(zip(self.joint_names, state))
        action_dict = dict(zip(self.joint_names, action))
        
        return {
            'state': state,
            'action': action,
            'state_dict': state_dict,
            'action_dict': action_dict,
            'frame_index': episode['frame_index'],
            'episode_index': episode['episode_index']
        }

    def execute_action(self, action):
        """Execute a single action on the robot"""
        # Split the action into body and hand components
        body_action = action[:14]
        left_hand_action = action[14:21]
        right_hand_action = action[21:]
        
        # Execute arm action
        self.arm_ctrl.ctrl_dual_arm(body_action, np.zeros(14))
        
        # Execute hand actions
        self.left_hand_array[:] = left_hand_action
        self.right_hand_array[:] = right_hand_action

    def run_episode(self, episode_index=0, delay=1/30.0):
        """Run a complete episode from the dataset"""
        print(f"\nRunning episode {episode_index}...")
        
        # Get the episode data
        episode = self.train_dataset[episode_index]
        
        # Get the initial state
        init_state = episode['observation.state']
        init_left_arm_pose = init_state[:14]
        init_left_hand_pose = init_state[14:21]
        init_right_hand_pose = init_state[21:]

        print("Initializing robot pose...")
        self.arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        self.left_hand_array[:] = init_left_hand_pose
        self.right_hand_array[:] = init_right_hand_pose
        
        print("Waiting for robot to reach initial pose...")
        time.sleep(2)

        # Get all actions for this episode
        actions = episode['action']
        if not isinstance(actions, list):
            actions = [actions]  # Convert single action to list if needed
            
        print(f"Replaying {len(actions)} actions...")
        
        # Replay each action
        for action_idx, action in enumerate(actions):
            # Execute the action
            self.execute_action(action)
            
            # Update timing information
            current_time = time.time()
            if self.last_step_time is not None:
                step_time = current_time - self.last_step_time
                self.total_execution_time += step_time
                self.step_count += 1
                if self.step_count % 10 == 0:  # Log every 10 steps
                    avg_time = self.total_execution_time / self.step_count
                    print(f"Action {action_idx + 1}/{len(actions)} - Average step time: {avg_time:.3f}s")
            self.last_step_time = current_time
            
            # Wait for next action
            time.sleep(delay)

        print(f"Completed episode {episode_index}")

    def run(self):
        """Main loop to run multiple episodes"""
        print("Starting dataset replayer...")
        print("Press Ctrl+C to exit")
        
        try:
            # Wait for user input to start
            user_input = input("Please enter the start signal (enter 's' to start replaying actions):")
            if user_input.lower() != 's':
                print("Exiting...")
                return

            episode_index = 0
            while True:
                self.run_episode(episode_index)
                episode_index = (episode_index + 1) % len(self.train_dataset)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Clean up shared memory
            self.tv_img_shm.close()
            self.tv_img_shm.unlink()
            self.wrist_img_shm.close()
            self.wrist_img_shm.unlink()

if __name__ == "__main__":
    replayer = DatasetReplayer()
    replayer.run() 