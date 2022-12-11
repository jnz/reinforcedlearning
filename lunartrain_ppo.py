import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# The key concept is that the control problem and physical world
# is simulated with an "env" object.
# The machine learning is done with the "model" object that is
# trained on the "env" object.
#
# The key function of the env object is a simulation step with control input in
# "action" argument with the step function:
#   env.step(action): returns observation, reward, done, info

# Create a vectorized environment (method for stacking multiple independent
# environments into a single environment)
env = make_vec_env('LunarLander-v2', n_envs=16)
# Alternative:
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

# Instanciate the agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=1024,
    learning_rate=5e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_lunar_tensorboard/",
    device="cuda")

# Train and save the model
model.learn(total_timesteps=10000000)
model_name = 'jan-ppo-lunar-lander-v7'
model.save(model_name)

# Test the trained model
eval_env = gym.make("LunarLander-v2")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
