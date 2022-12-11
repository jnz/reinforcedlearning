import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load the PPO model from file
model_name = 'jan-ppo-lunar-lander-v6'
model = PPO.load(model_name)

# Create environment
eval_env = gym.make("LunarLander-v2")

# Run the model in the environment
obs = eval_env.reset()
cumulative_reward = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()
    cumulative_reward = cumulative_reward + rewards
    print(f"Step {i}: action={action}, reward={rewards}, done={dones}, info={info}")
    if dones == True:
        break

print(f"Cumulative reward: {cumulative_reward}")

# wait until user presses enter then close the window
input("Press Enter to close the window")

eval_env.reset()
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
