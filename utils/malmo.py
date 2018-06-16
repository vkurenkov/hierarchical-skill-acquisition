import torchvision.transforms as transforms

from PIL import Image

def preprocess_frame(frame):
    '''
    input:
        frame - a timestamped frame
    output:
        frame - a float tensor of size (3, frame.width, frame.height)
    '''
    frame = Image.frombytes("RGB", (frame.width, frame.height), bytes(frame.pixels))

    return transforms.ToTensor()(frame)

def wait_for_observations(agent_host):
    '''
    output:
        world_state - received world state
        observations - received observations
        is_running - whether the mission is on or not
    '''
    world_state = agent_host.getWorldState()
    observations = world_state.observations
    while world_state.number_of_observations_since_last_state == 0:
        if(not world_state.is_mission_running):
            break
        world_state = agent_host.getWorldState()
        observations = world_state.observations

    return world_state, observations, world_state.is_mission_running

def wait_for_frames(agent_host):
    '''
    output:
        world_state - received world state
        frames - received frames
        is_running - whether the mission is on or not
    '''
    world_state = agent_host.getWorldState()
    frames = world_state.video_frames
    while world_state.number_of_video_frames_since_last_state == 0:
        if(not world_state.is_mission_running):
            break
        world_state = agent_host.getWorldState()
        frames = world_state.video_frames

    return world_state, frames, world_state.is_mission_running