from pylab import *
from drawnow import drawnow
from utils.training import Session

import random
import time

session = Session("hsa_terminal_policy")
session.switch_group(1)

rewards = session.get_rewards()[:, 0]
timesteps = session.get_timesteps()[:, 0]
value_losses = session.get_value_losses()[:, 0]
a2c_losses = session.get_a2c_losses()[:, 0]
total_losses = session.get_total_losses()[:, 0]
entropies = session.get_actions_entropies()[:, 0]

ticks = []
mean_rewards = []
mean_timesteps = []
mean_value_losses = []
mean_a2c_losses = []
mean_total_losses = []
mean_entropies = []

n = 50
start = 0
end = start + n
while end < len(rewards):
    mean_rewards.append(np.mean(rewards[start:end]))
    mean_timesteps.append(np.mean(timesteps[start:end]))
    mean_value_losses.append(np.mean(value_losses[start:end]))
    mean_a2c_losses.append(np.mean(a2c_losses[start:end]))
    mean_total_losses.append(np.mean(total_losses[start:end]))
    mean_entropies.append(np.mean(entropies[start:end]))

    if len(ticks) > 0:
        ticks.append(ticks[len(ticks) - 1] + n)
    else:
        ticks.append(n)

    start = end
    end = start + n

figure(figsize=(4, 4))

def draw_figures():
    #subplot(1, 1, 1)
    figure(figsize=(4, 4))
    ylabel("Reward")
    ylim((0.0, 1.0))
    yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plot(ticks, mean_rewards, linestyle="dashed")
    show()

    # subplot(6, 1, 2)
    figure(figsize=(4, 4))
    ylabel("Timesteps")
    ylim((0, 35))
    plot(mean_timesteps, linestyle="dashed")
    show()

    # subplot(6, 1, 3)
    figure(figsize=(4, 4))
    ylabel("Value Loss")
    plot(mean_value_losses, linestyle="dashed")
    show()

    # subplot(6, 1, 4)
    figure(figsize=(4, 4))
    ylabel("A2C Loss")
    plot(mean_a2c_losses, linestyle="dashed")
    show()

    # subplot(6, 1, 5)
    figure(figsize=(4, 4))
    ylabel("Total Loss")
    plot(mean_total_losses, linestyle="dashed")
    show()

    # subplot(6, 1, 6)
    figure(figsize=(4, 4))
    xlabel("Episodes")
    ylabel("Entropy")
    plot(mean_entropies, linestyle="dashed")
    show()

draw_figures()
while True:
    time.sleep(10)
# while True:
#     drawnow(draw_figures, stop_on_close=True, show_once=True)
#     time.sleep(0.1)
