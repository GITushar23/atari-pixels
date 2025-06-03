import os
import cv2
import numpy as np
import torch
import gym
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from dqn_agent import DQNAgent
from random_agent import RandomAgent
import argparse
import shutil
import tempfile

class MarioRandomAgent:
    """Random agent for Super Mario Bros with 7 actions"""
    def __init__(self, n_actions=7):
        self.n_actions = n_actions
    
    def select_action(self, temperature=1.0):
        return np.random.randint(0, self.n_actions)

def create_mario_env():
    """Create and return a Super Mario Bros environment"""
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def record_mario_episode_no_overlay(env, agent, video_writer, temperature=1.0, frame_size=None, debug=False, png_dir=None):
    """Record a single Mario episode without overlay"""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new gym API
    
    # Initialize state stack for DQN
    if hasattr(agent, 'policy_net'):  # DQN agent
        # Convert to grayscale and resize for DQN
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(gray_obs, (84, 84))
        state_stack = np.stack([resized_obs] * 8, axis=0)
    
    frame_num = 0
    max_steps = 2000  # Mario episodes can be longer
    
    while True:
        if frame_num > max_steps:
            if debug:
                print(f"Max steps ({max_steps}) reached, ending episode")
            break
        
        # Get the current frame for video
        try:
            # The observation from Mario is already the RGB frame we want
            raw_frame = obs
            if debug and frame_num % 100 == 0:
                print(f"Frame {frame_num} shape: {raw_frame.shape}")
        except Exception as e:
            if debug:
                print(f"Error getting frame {frame_num}: {e}")
            raw_frame = np.zeros((240, 256, 3), dtype=np.uint8)  # Mario frame size
        
        if raw_frame is None or raw_frame.size == 0:
            if debug:
                print(f"Warning: Empty frame at step {frame_num}")
            raw_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        
        # Resize frame if needed for video writer
        if frame_size is not None and hasattr(video_writer, 'write'):
            expected_width, expected_height = frame_size[0], frame_size[1]
            actual_height, actual_width = raw_frame.shape[0], raw_frame.shape[1]
            
            if actual_height != expected_height or actual_width != expected_width:
                if expected_width > 0 and expected_height > 0:
                    raw_frame = cv2.resize(raw_frame, (expected_width, expected_height))
        
        # Write frame to video
        try:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
        except Exception as e:
            if debug:
                print(f"Error writing frame {frame_num}: {e}")
        
        # Save frame as PNG if png_dir is provided
        if png_dir is not None:
            png_path = os.path.join(png_dir, f"{frame_num}.png")
            try:
                # Save as RGB
                cv2.imwrite(png_path, cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                if debug:
                    print(f"Error saving PNG frame {frame_num}: {e}")
        
        # Get action from agent
        if hasattr(agent, 'policy_net'):  # DQN agent
            if temperature is None:
                action = agent.select_action(state_stack, mode='greedy')
            else:
                action = agent.select_action(state_stack, mode='softmax', temperature=temperature)
        else:  # Random agent
            action = agent.select_action(temperature=temperature)
        
        # Take step in environment
        step_result = env.step(action)
        if len(step_result) == 4:  # Old gym API
            next_obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:  # New gym API
            next_obs, reward, terminated, truncated, info = step_result
        
        obs = next_obs
        frame_num += 1
        
        # Update state stack for DQN
        if hasattr(agent, 'policy_net'):  # DQN agent
            gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized_obs = cv2.resize(gray_obs, (84, 84))
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = resized_obs
        
        # Check if episode is done
        if terminated or truncated or info.get('time', 400) < 10:
            if debug:
                print(f"Episode ended at frame {frame_num}. Terminated: {terminated}, Truncated: {truncated}, Time: {info.get('time', 'N/A')}")
            break
    
    if debug:
        print(f"Episode completed - recorded {frame_num} frames")
    
    return frame_num

def record_bulk_mario_videos(
    total_videos=10,
    output_dir='mario_videos',
    percent_random=50,
    skill_level=None,
    no_sync=False,
    temperature=1.0,
    debug=False,
    frame_size=None,
    fps=30
):
    """Bulk generate Mario videos without text overlay, with a specified percentage from random agent."""
    temp_dir = None
    working_dir = output_dir
    
    if no_sync:
        temp_dir = tempfile.mkdtemp()
        working_dir = temp_dir
        print(f"Using temporary directory: {temp_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Create environment and agents
    env = create_mario_env()
    random_agent = MarioRandomAgent(n_actions=7)
    dqn_agent = DQNAgent(n_actions=7, state_shape=(8, 84, 84))
    
    # Load DQN model if available
    if skill_level is not None:
        model_path = os.path.join('checkpoints', f'dqn_skill_{skill_level}.pth')
        checkpoint_name = f'skill_{skill_level}'
    else:
        model_path = os.path.join('checkpoints', 'dqn_latest.pth')
        checkpoint_name = 'latest'
    
    if os.path.exists(model_path):
        dqn_agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
        dqn_agent.policy_net.eval()
        print(f"Model loaded from {model_path}. Device: {next(dqn_agent.policy_net.parameters()).device}")
    else:
        print(f"Warning: No checkpoint found at {model_path}, using untrained model")

    # Get frame size if not provided
    if frame_size is None:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if obs is not None:
            frame_height, frame_width = obs.shape[:2]
            frame_size = (frame_width, frame_height)
        else:
            frame_size = (256, 240)  # Default Mario frame size
    
    print(f"[Bulk] Using frame size: {frame_size}")

    # Test video codecs
    codecs = [
        ('MJPG', '.avi'),
        ('XVID', '.avi'),
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('FMP4', '.mp4'),
        ('H264', '.mp4'),
        ('X264', '.mp4'),
    ]
    
    working_codec = None
    for codec, ext in codecs:
        try:
            test_path = os.path.join(working_dir, f'test{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size)
            if not test_writer.isOpened():
                continue
            
            # Test with a dummy frame
            dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            test_writer.write(dummy_frame)
            test_writer.release()
            
            if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
                working_codec = (codec, ext)
                os.remove(test_path)
                break
            if os.path.exists(test_path):
                os.remove(test_path)
        except Exception:
            continue
    
    if working_codec is None:
        print("Error: Could not find a working video codec. Using MJPG as fallback.")
        working_codec = ('MJPG', '.avi')
    
    fourcc = cv2.VideoWriter_fourcc(*working_codec[0])
    file_ext = working_codec[1]
    print(f"[Bulk] Using codec: {working_codec[0]} with extension: {file_ext}")

    # Determine number of random and trained videos
    num_random = int(total_videos * percent_random / 100)
    num_trained = total_videos - num_random
    print(f"[Bulk] Generating {num_random} random agent videos and {num_trained} trained agent videos.")

    # Record random agent videos
    for i in range(num_random):
        video_path = os.path.join(working_dir, f'bulk_random_agent_{i+1}{file_ext}')
        png_dir = os.path.join(working_dir, f'bulk_random_agent_{i+1}')
        os.makedirs(png_dir, exist_ok=True)
        
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            continue
        
        try:
            frames_recorded = record_mario_episode_no_overlay(
                env, random_agent, video_writer, 
                temperature=temperature, frame_size=frame_size, 
                debug=debug, png_dir=png_dir
            )
            print(f"[Bulk] Random agent video {i+1} saved: {video_path} ({frames_recorded} frames) and frames in {png_dir}")
        except Exception as e:
            print(f"Error recording random agent video {i+1}: {e}")
        finally:
            video_writer.release()

    # Record trained agent videos
    for i in range(num_trained):
        video_path = os.path.join(working_dir, f'bulk_trained_agent_{checkpoint_name}_{i+1}{file_ext}')
        png_dir = os.path.join(working_dir, f'bulk_trained_agent_{checkpoint_name}_{i+1}')
        os.makedirs(png_dir, exist_ok=True)
        
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            continue
        
        try:
            frames_recorded = record_mario_episode_no_overlay(
                env, dqn_agent, video_writer, 
                temperature=temperature, frame_size=frame_size, 
                debug=debug, png_dir=png_dir
            )
            print(f"[Bulk] Trained agent video {i+1} saved: {video_path} ({frames_recorded} frames) and frames in {png_dir}")
        except Exception as e:
            print(f"Error recording trained agent video {i+1}: {e}")
        finally:
            video_writer.release()

    # Clean up environment
    try:
        env.close()
    except Exception:
        pass
    
    try:
        del env
    except Exception:
        pass

    # Copy files from temp directory if using no_sync
    if no_sync:
        print(f"\nCopying files from temporary directory to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        for item in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, item)
            dst_path = os.path.join(output_dir, item)
            
            if os.path.isdir(src_path):
                # Copy directory
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied directory: {item}")
            elif os.path.getsize(src_path) > 1000:
                # Copy file if it's not corrupted
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {item} ({os.path.getsize(dst_path)} bytes)")
            else:
                print(f"Skipped corrupted file: {item}")
        
        shutil.rmtree(temp_dir)

    print("\n[Bulk] All Mario videos generated successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record Super Mario Bros gameplay videos with random and trained agents')
    parser.add_argument('--output_dir', type=str, default='mario_videos', help='Directory to save videos')
    parser.add_argument('--skill_level', type=int, choices=[0,1,2,3], help='Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.')
    parser.add_argument('--no_sync', action='store_true', help='Use a temporary directory to avoid iCloud sync issues')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for agent action selection (default: 1.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for troubleshooting video recording issues')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for video recording')
    
    # Bulk generation arguments
    parser.add_argument('--total_videos', type=int, default=10, help='Total number of videos to generate')
    parser.add_argument('--percent_random', type=float, default=50, help='Percentage of random agent videos (0-100)')
    
    args = parser.parse_args()
    
    record_bulk_mario_videos(
        total_videos=args.total_videos,
        output_dir=args.output_dir,
        percent_random=args.percent_random,
        skill_level=args.skill_level,
        no_sync=args.no_sync,
        temperature=args.temperature,
        debug=args.debug,
        fps=args.fps
    )