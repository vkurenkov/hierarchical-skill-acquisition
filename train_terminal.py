import MalmoPython
import os
import random
import env
import datetime
import time
import torch
import numpy as np

from collections import namedtuple
from torch.autograd import Variable
from torch.distributions import Categorical
from agent.hierarchical.terminal import TerminalPolicy
from experience import ExperienceReplay, Memento
from utils.malmo import wait_for_frames, wait_for_observations, preprocess_frame
from utils.training import Session

# Initialize agent host
agent_host = MalmoPython.AgentHost()
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)


# Initialize training session
session = Session("hsa_terminal_policy_scenario_2.0")
session.switch_group()

# Training constants
TASKS = [("Find", "Red"), ("Find", "Yellow"), ("Find", "Blue"), 
         ("Find", "Green"), ("Find", "White"), ("Find", "Black")]
BATCH_SIZE = 36
SEED = 0
SEED_REPETITIONS = 5
SAVE_EVERY = 100
NUM_LOW_LEVEL_ACTIONS = 6
NUM_PAST_TIMESTEPS = 4
max_EPISODE_LENGTH = 50
DISCOUNT = 0.95
REPLAY_SIZE = 800
DESCRIPTION = "Sparse reward; Ordinary experience replay; Remove entropy from total loss;"
EPSILON_START = 0.9
EPSILON_DECAY = 0.00005 # Reach minimum in about 20k episodes
EPSILON_MINIMUM = 0.00
REWARD_THERSHOLD = 0.9
VOCABULARY = {"Find": 0, "Red": 1, "Yellow": 2, "Blue": 3, "Green": 4, "White": 5, "Black": 6}

# Initialize terminal policy
torch.manual_seed(SEED)
policy = TerminalPolicy(num_actions=NUM_LOW_LEVEL_ACTIONS, 
                        num_timesteps=NUM_PAST_TIMESTEPS,
                        vocabulary_size=len(VOCABULARY))

# Write information about current training session
session.log("\nTraining Session at " + str(datetime.datetime.now()))
session.log("\nDescription: " + DESCRIPTION)
session.log("\nTasks to learn: ")
for task in TASKS:
    session.log(task[0] + " " + task[1] + "; ")
session.log("\n")
session.log("\nEpisode length: " + str(max_EPISODE_LENGTH) + "; Batch size: " + str(BATCH_SIZE))
session.log("\nNum low-level actions: " + str(NUM_LOW_LEVEL_ACTIONS) + "; Num memory timesteps: " + str(NUM_PAST_TIMESTEPS))
session.log("\nReplay size: " + str(REPLAY_SIZE) + "; Return Discount: " + str(DISCOUNT))
session.log("\nEpsilon Start: " + str(EPSILON_START) + "; Epsilon Decay: " + str(EPSILON_DECAY) + "; Epsilon Min: " + str(EPSILON_MINIMUM))
session.log("\nSeed: " + str(SEED))


for task in TASKS:
    session.log("\n---> Training for subtask: " + str(task[0]) + " " + str(task[1]))

    # Define instruction early
    instruction = torch.LongTensor([VOCABULARY[task[0]], VOCABULARY[task[1]]]).unsqueeze(0)

    # Train several times starting from different seeds
    for seed in range(SEED, SEED + SEED_REPETITIONS):
        session.log("\n---> Start new seed=" + str(seed))

        replay_memory = ExperienceReplay(capaciy=REPLAY_SIZE)
        episode_num = 0
        epsilon = EPSILON_START

        while True:
            episode_num += 1

            # Notify about the episode we start
            session.log("\n---------> Episode #" + str(episode_num))
            session.log("; Start at: " + str(datetime.datetime.now()) + "; ")
            session.log_flush()

            # Load training environment
            my_mission = env.create_environment(task, seed=seed+episode_num)
            random.seed(seed + episode_num)

            # Start the environment
            max_retries = 3
            my_mission_record = MalmoPython.MissionRecordSpec()
            for retry in range(max_retries):
                try:
                    agent_host.startMission( my_mission, my_mission_record )
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        print("Error starting mission:",e)
                        exit(1)
                    else:
                        time.sleep(2.5)

            print("Waiting for the mission to start", end=' ')
            world_state = agent_host.getWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:",error.text)

            # Wait for the first observation to arrive...
            print("\nWaiting for the first observation...")
            world_state, cur_observation, _ = wait_for_observations(agent_host)

            # Then we wait for the first frame to arrive...
            print("Waiting for the first frame...")
            world_state, cur_frames, _ = wait_for_frames(agent_host)

            # Now the episode really starts
            print("The mission has started...")

            Point = namedtuple("Point", "frames instruction action reward action_probs")
            episode_trajectory = []
            num_timesteps = 0

            cur_frame = cur_frames[0]
            memory_frames = [cur_frame for _ in range(NUM_PAST_TIMESTEPS)]
            while world_state.is_mission_running:
                # Save the previous observation
                prev_observation = cur_observation
                memory_frames.insert(0, cur_frame)
                memory_frames.pop()
                
                # Sample an action from the policy
                last_frames = preprocess_frame(memory_frames[0]).unsqueeze(0)
                for frame in memory_frames[1:]:
                    last_frames = torch.cat((last_frames, preprocess_frame(frame).unsqueeze(0)), 0)

                action_probs = policy.forward(Variable(last_frames.unsqueeze(0)), Variable(instruction))
                top_action = torch.max(action_probs, 1)[1]
                prob_choose_top = 1.0 - epsilon
                action_probs[0, :] = action_probs[0, :] * (1 - prob_choose_top)
                action_probs[0, top_action.data[0]] = action_probs[0, top_action.data[0]] + prob_choose_top
                
                sampled_action = Categorical(action_probs).sample()

                print(action_probs)
                print(epsilon)

                # Act and update the current observation
                cur_observation, cur_frame, done = env.act(agent_host, sampled_action.data[0])  
                num_timesteps += 1

                if(done):
                    break

                # Calculate the reward based on the current task
                reward = env.get_reward(task, cur_observation, prev_observation)

                # Check for episode's termination
                if(reward > 0):
                    done = True
                if(num_timesteps >= max_EPISODE_LENGTH):
                    done = True
                    if(reward < 1.0):
                        reward = -1.0
                
                episode_trajectory.append(Point(last_frames.numpy(), instruction.numpy(), sampled_action.data.numpy(), reward, action_probs.data.numpy()))

                if(done):
                    break

            print("The mission has ended.")
            epsilon -= EPSILON_DECAY
            epsilon = max(EPSILON_MINIMUM, epsilon)
            session.log("End at: " + str(datetime.datetime.now()) + "; ")
            session.log("Reward: " + str(reward) + "; Num Timesteps: " + str(num_timesteps) + "; ")

            agent_host.sendCommand("quit")

            print("Start the training process...")

            # Check if mean reward is higher than we wanted it to be
            session.reward(reward)
            session.timesteps(num_timesteps)
            if(episode_num % SAVE_EVERY == 0):
                mean_reward = np.mean(session.get_rewards()[-SAVE_EVERY:, 0])
                mean_timesteps = np.mean(session.get_timesteps()[-SAVE_EVERY, 0])
                mean_entropy = np.mean(session.get_actions_entropies()[-SAVE_EVERY+1, 0])
                session.log("\n---------> Mean Reward: " + str(mean_reward))
                session.log("; Mean Timesteps: " + str(mean_timesteps))
                session.log("; Mean Entropy: " + str(mean_entropy))
                print("Save the model...")
                session.checkpoint_model(policy, task[0] + "_" + task[1] + "_" + str(seed) + "_" + str(episode_num))

                if(mean_reward >= REWARD_THERSHOLD):
                    session.log("\n---------> Mean Reward achieved required threshold. Success!")
                    break

            # Calculate episode returns and put the episode trajectory into the replay memory
            value_return = 0.0
            for point in reversed(episode_trajectory):
                value_return = point.reward + DISCOUNT * value_return
                replay_memory.append(Memento(point.frames, point.instruction,
                                            point.action, value_return, 
                                            point.action_probs))

            for i in range(random.randint(1, 4)):
                # Train the model
                mementos = replay_memory.sample(BATCH_SIZE)
                mementos_frames = Variable(torch.FloatTensor(mementos[0].frames).unsqueeze(0))
                mementos_instructions = Variable(torch.LongTensor(mementos[0].instruction))
                mementos_actions = Variable(torch.LongTensor(mementos[0].action).unsqueeze(0))
                mementos_returns = Variable(torch.FloatTensor([mementos[0].value_return]).unsqueeze(0))
                mementos_probs = Variable(torch.FloatTensor(mementos[0].action_probs))
                for memento in mementos[1:]:
                    mementos_frames = torch.cat((mementos_frames, Variable(torch.FloatTensor(memento.frames).unsqueeze(0))), 0)
                    mementos_instructions = torch.cat((mementos_instructions, Variable(torch.LongTensor(memento.instruction))), 0)
                    mementos_actions = torch.cat((mementos_actions, Variable(torch.LongTensor(memento.action).unsqueeze(0))), 0)
                    mementos_returns = torch.cat((mementos_returns, Variable(torch.FloatTensor([memento.value_return]).unsqueeze(0))), 0)
                    mementos_probs = torch.cat((mementos_probs, Variable(torch.FloatTensor(memento.action_probs))), 0)

                total_loss, value_loss, a2c_loss, entropy = policy.train(mementos_frames, mementos_instructions, 
                                    mementos_actions, mementos_returns, mementos_probs)

            # Report current losses
            session.log("; Entropy: " + str(entropy.data[0]))
            session.log("; Total Loss: " + str(total_loss.data[0]))
            session.log("; Value Loss: " + str(value_loss.data[0]))
            session.log("; A2C Loss: " + str(a2c_loss.data[0]))

            session.actions_entropy(entropy.data[0])
            session.a2c_loss(a2c_loss.data[0])
            session.value_loss(value_loss.data[0])
            session.total_loss(total_loss.data[0])

session.close()