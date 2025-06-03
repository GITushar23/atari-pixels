import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any
import ale_py
gym.register_envs(ale_py)
class AtariBreakoutEnv:
    """
    Wrapper for Atari Breakout environment with preprocessing.
    If return_rgb=True, returns original RGB frames (210, 160, 3) instead of preprocessed grayscale (84, 84).
    """
    
    def __init__(self, return_rgb: bool = False):
        """Initialize the environment with specific settings.
        Args:
            return_rgb: If True, return original RGB frames in reset/step.
        """
        # Create the base Atari environment with specific settings
        self.env = gym.make(
            "ALE/Breakout-v5",
            render_mode='rgb_array',  # Enable rgb_array rendering
            frameskip=4,  # Skip 4 frames between actions
            repeat_action_probability=0.0,  # Disable sticky actions
            full_action_space=False  # Use minimal action space
        )
        
        # Store action space size for the agent
        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}")
        print(f"Action meanings: {self.env.unwrapped.get_action_meanings()}")
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
        
        # Track lives for monitoring
        self.lives = 5  # Breakout starts with 5 lives
        self.was_real_done = True  # Track if the episode was actually done
        
        # Reward shaping configuration
        self.living_penalty = -0.005  # Small negative reward per time step
        self.life_loss_penalty = 0.0  # Substantial penalty for losing a life
        
        self.return_rgb = return_rgb
    
    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert RGB observation to 84x84 grayscale."""
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation.
        Returns original RGB if self.return_rgb else preprocessed grayscale.
        """
        obs, info = self.env.reset()
        self.lives = info.get('lives', 5)  # Get initial lives
        
        if self.return_rgb:
            return obs, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        Returns original RGB if self.return_rgb else preprocessed grayscale.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track lives to detect life loss
        lives = info.get('lives', 0)
        life_loss_reward = 0.0
        if lives < self.lives:
            # Lost a life - apply penalty
            life_loss_reward = self.life_loss_penalty
            self.lives = lives
        
        # Apply living penalty and life loss penalty
        shaped_reward = reward + self.living_penalty + life_loss_reward
        #shaped_reward = reward
        
        if self.return_rgb:
            return obs, shaped_reward, terminated, truncated, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, shaped_reward, terminated, truncated, info
    
    def close(self):
        """Close the environment."""
        self.env.close() 



"""NEW NETHOD"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT # Or another action space
from nes_py.wrappers import JoypadSpace



class SuperMarioBrosEnv:
    def __init__(self, rom_mode='SuperMarioBros-v3', return_rgb: bool = False, new_width=84, new_height=84): # Add new_width, new_height for DQN
        # Use apply_api_compatibility for Gymnasium interface
        self.env = gym_super_mario_bros.make(rom_mode)
        # Wrap with action space simplifier
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT) # SIMPLE_MOVEMENT gives 7 discrete actions

        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}") # Will be Discrete(7) for SIMPLE_MOVEMENT
        # print(f"Action meanings: {SIMPLE_MOVEMENT}") # To see what actions correspond to

        self.new_width = new_width
        self.new_height = new_height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.new_height, self.new_width), dtype=np.uint8 # For grayscale DQN
        )
        self.return_rgb = return_rgb # For VQ-VAE data collection later

        # Mario-specific reward shaping parameters (example)
        self.x_pos_reward_scale = 0.01
        self.time_penalty_scale = -0.001
        self.death_penalty = -1.0
        self.flag_reward = 5.0

        self.current_x_pos = 0
        self.current_score = 0
        self.current_time = 400 # Default starting time
        self.current_lives = 2 # Default starting lives in SMB is 2 (meaning 3 total tries)

    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        # Input obs is HxWxC (e.g., 240x256x3)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA)
        return resized # (new_height, new_width)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self.env.reset()
        # Update initial stats from info
        self.current_x_pos = 0
        self.current_score = 0
        self.current_time = 400
        self.current_lives = 2 # 'life' in gym-super-mario-bros info

        if self.return_rgb:
            return obs ,None# Return raw 240x256x3 frame
        processed_obs = self.preprocess_observation(obs)
        return processed_obs ,None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)

        # --- Mario Specific Reward Shaping ---
        shaped_reward = float(reward) # Start with base game reward

        # Reward for moving right
        x_pos_change = info.get('x_pos', self.current_x_pos) - self.current_x_pos
        shaped_reward += x_pos_change * self.x_pos_reward_scale
        self.current_x_pos = info.get('x_pos', self.current_x_pos)

        # Penalty for time passing (if time decreases)
        time_change = info.get('time', self.current_time) - self.current_time
        if time_change < 0: # Time decreased
            shaped_reward += time_change * self.time_penalty_scale # time_change is negative
        self.current_time = info.get('time', self.current_time)

        # Penalty for dying
        if info.get('life', self.current_lives) < self.current_lives:
            shaped_reward += self.death_penalty
        self.current_lives = info.get('life', self.current_lives)
        
        # Reward for reaching flag
        if info.get('flag_get', False):
            shaped_reward += self.flag_reward

        # Update score
        self.current_score = info.get('score', self.current_score)

        if self.return_rgb:
            return obs, shaped_reward,done,done, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, shaped_reward, done,done, info

    def close(self):
        self.env.close()