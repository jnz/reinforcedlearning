import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Create a vectorized environment (method for stacking multiple independent
# environments into a single environment)
env = make_vec_env('LunarLander-v2', n_envs=16)
# env = gym.make("LunarLander-v2")

observation = env.reset()

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation
# Observations:
# - Horizontal pad coordinate (x)
# - Vertical pad coordinate (y)
# - Horizontal speed (x)
# - Vertical speed (y)
# - Angle
# - Angular speed
# - If the left leg has contact point touched the land
# - If the right leg has contact point touched the land

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action
# Actions:
# - Do nothing,
# - Fire left orientation engine,
# - Fire the main engine,
# - Fire right orientation engine.

# Reward:
# - Moving from the top of the screen to the landing pad and zero speed is about 100~140 points.
# - Firing main engine is -0.3 each frame
# - Each leg ground contact is +10 points
# - Episode finishes if the lander crashes (additional - 100 points) or come to rest (+100 points)

# Instanciate the model
model = DQN(
    "MlpPolicy", env=env, verbose=1,
    device="cuda")

# Train and save the model
model.learn(total_timesteps=2000000)
model_name = 'jan-ppo-lunar-lander-v2_dqn_2M'
model.save(model_name)
