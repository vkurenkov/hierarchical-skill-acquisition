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
from utils.malmo import wait_for_frames, wait_for_observations, preprocess_frame

# Initialize agent host
agent_host = MalmoPython.AgentHost()
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)


# Initialize training logger
logger = open(os.path.join(".", "logs", "training_policy_k=0.log"), "a")

# Training constants
TASKS = [("Find", "Red"), ("Find", "Yellow"), ("Find", "Blue"), 
         ("Find", "Green"), ("Find", "White"), ("Find", "Black")]
BATCH_SIZE = 32
NUM_LOW_LEVEL_ACTIONS = 8
NUM_PAST_TIMESTEPS = 4
max_EPISODE_LENGTH = 50
DISCOUNT = 0.95
REWARD_THERSHOLD = 0.9
VOCABULARY = {"Find": 0, "Red": 1, "Yellow": 2, "Blue": 3, "Green": 4, "White": 5, "Black": 6}

# Initialize terminal policy
policy = TerminalPolicy(num_actions=NUM_LOW_LEVEL_ACTIONS, 
                        num_timesteps=NUM_PAST_TIMESTEPS,
                        vocabulary_size=len(VOCABULARY))

# Write information about current training session
logger.write("\nTraining Session at " + str(datetime.datetime.now()))
logger.write("\nEpisode length: " + str(max_EPISODE_LENGTH) + "; Batch size: " + str(BATCH_SIZE))
logger.write("\nNum low-level actions: " + str(NUM_LOW_LEVEL_ACTIONS) + "; Num memory timesteps: " + str(NUM_PAST_TIMESTEPS))
logger.write("\nTasks to learn: ")
for task in TASKS:
    logger.write(task[0] + " " + task[1] + "; ")
logger.write("\n")


for task in TASKS:
    logger.write("\n---> Training for subtask: " + str(task[0]) + " " + str(task[1]))

    Memento = namedtuple("Memento", "frames instruction action value_return action_probs")
    replay_memory = []
    last_200_rewards = []
    episode_num = 0

    while True:
        episode_num += 1

        # Notify about the episode we start
        logger.write("\n---------> Episode #" + str(episode_num))
        logger.write("; Start at: " + str(datetime.datetime.now()) + "; ")
        logger.flush()

        # Load training environment
        my_mission = env.create_environment(task, seed=episode_num)

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

        # Define instruction early
        instruction = Variable(torch.LongTensor([VOCABULARY[task[0]], VOCABULARY[task[1]]]).unsqueeze(0))

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
            action_probs = policy.forward(Variable(last_frames.unsqueeze(0)), instruction)
            sampled_action = Categorical(action_probs).sample()

            # Act and update the current observation
            cur_observation, cur_frame, done = env.act(agent_host, sampled_action.data[0])  
            num_timesteps += 1

            # Calculate the reward based on the current task
            reward = env.get_reward(task, cur_observation, prev_observation)

            episode_trajectory.append(Point(Variable(last_frames), instruction, sampled_action, 
                                            reward, action_probs))

            # Check for episode's termination
            if(reward > 0):
                done = True
            if(num_timesteps >= max_EPISODE_LENGTH):
                done = True

            if(done):
                break

        print("The mission has ended.")
        logger.write("End at: " + str(datetime.datetime.now()) + "; ")
        logger.write("Reward: " + str(reward) + "; Num Timesteps: " + str(num_timesteps) + "; ")

        agent_host.sendCommand("quit")

        print("Start the training process...")

        # Save the reward
        # Save number of timesteps

        # Check if mean reward is higher than we wanted it to be
        last_200_rewards.append(reward)
        if(episode_num % 200 == 0):
            mean_reward = np.mean(last_200_rewards)
            print("Mean reward for last 200 episodes: " + str(mean_reward))
            logger.write("\n---------> Mean Reward: " + str(mean_reward))
            last_200_rewards = []

            if(mean_reward >= REWARD_THERSHOLD):
                logger.write("\n---------> Mean Reward achieved required threshold. Success!")
                break

        # Calculate episode returns and put the episode trajectory into the replay memory
        value_return = 0.0
        for point in reversed(episode_trajectory):
            value_return += point.reward + DISCOUNT * value_return
            replay_memory.append(Memento(point.frames, point.instruction,
                                         point.action, Variable(torch.FloatTensor([value_return])), 
                                         point.action_probs))
        
        # Train the model
        if(BATCH_SIZE <= len(replay_memory)):
            mementos = random.sample(replay_memory, BATCH_SIZE)
        else:
            mementos = random.sample(replay_memory, len(replay_memory))
        mementos_frames = mementos[0].frames.unsqueeze(0)
        mementos_instructions = mementos[0].instruction
        mementos_actions = mementos[0].action.unsqueeze(0)
        mementos_returns = mementos[0].value_return.unsqueeze(0)
        mementos_probs = mementos[0].action_probs
        for memento in mementos[1:]:
            mementos_frames = torch.cat((mementos_frames, memento.frames.unsqueeze(0)), 0)
            mementos_instructions = torch.cat((mementos_instructions, memento.instruction), 0)
            mementos_actions = torch.cat((mementos_actions, memento.action.unsqueeze(0)), 0)
            mementos_returns = torch.cat((mementos_returns, memento.value_return.unsqueeze(0)), 0)
            mementos_probs = torch.cat((mementos_probs, memento.action_probs), 0)

        loss = policy.train(mementos_frames, mementos_instructions, 
                            mementos_actions, mementos_returns, mementos_probs)
        # Report the current loss
        print("Current loss: " + str(loss.data[0]))
        logger.write("Loss: " + str(loss.data[0]))

        # Checkpoint the model every 100 or 200 episodes

logger.close()