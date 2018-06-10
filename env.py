import MalmoPython
import os
import random
import numpy as np
import warnings
import xml.etree.ElementTree as ET

TRAIN_ENV_PATH = os.path.join(".", "train_env.xml")
TEST_ENV_PATH = os.path.join(".", "test_env.xml")

INSTRUCTIONS = ["Find", "Get", "Put", "Stack"]
OBJECTS = ["Red", "Blue", "Green", "White", "Yellow", "Black"]
TASKS = [(ins, obj) for ins in INSTRUCTIONS for obj in OBJECTS]

# No more than 20
NUM_BLOCKS_TO_GENERATE = 6
MAXIMUM_ROUNDS = 2000

# Relative positions of the free blocks on the train grid
TRAIN_FREE_GRID = [
    (x, z) for z in range(4) for x in range(6)
]
TRAIN_FREE_GRID.extend(
    [(x, z) for z in range(5, 8) for x in range(6)]
)
TRAIN_FREE_GRID.extend([(4, 4), (5, 4)])
TRAIN_ROOM1_START = (1, 1)
TRAIN_ROOM2_START = (8, 1)


def create_environment(task, train=True, seed=0):
    '''
    Creates a malmo's mission specification based on the given task

    task - a tuple (instruction: string, object: string)
    train - determines which environment to use (please, refer to the original paper)
    '''
    random.seed(seed)
    np.random.seed(seed)

    env_file = TRAIN_ENV_PATH
    if(not train):
        warnings.warn("Test environment is not currently supported. Switch to the training one.")

    print("Loading enviornment description from %s" % env_file)
    tree = ET.parse(env_file)
    namespaces = {"malmo": "http://ProjectMalmo.microsoft.com"}
    drawing_decorator = tree.getroot() \
                        .find("malmo:ServerSection", namespaces) \
                        .find("malmo:ServerHandlers", namespaces) \
                        .find("malmo:DrawingDecorator", namespaces)
    
    # Choose the room
    room_start = TRAIN_ROOM1_START
    if(random.random() > 0.5):
        room_start = TRAIN_ROOM2_START

    # All of the grid cells are free
    free_grid_pool = set(TRAIN_FREE_GRID)

    # Sample agent's position
    agent_position = random.sample(free_grid_pool, 1)[0]
    free_grid_pool.remove(agent_position)

    # Sample colours to generate
    instruction = task[0]
    colour = task[1]
    blocks_to_generate = [colour]
    if(instruction == "Stack"):
        blocks_to_generate.append(colour)
    blocks_to_generate.extend(np.random.choice(OBJECTS, NUM_BLOCKS_TO_GENERATE - len(blocks_to_generate), replace=True))

    # Randomly place blocks on the grid
    # But all of the blocks must be reachable from the agent's initial position
    print("Sampling block positions.", end="")
    num_generated = 0
    num_rounds = 0
    block_positions = []
    while(num_generated < NUM_BLOCKS_TO_GENERATE):
        if(len(free_grid_pool) == 0):
            if(num_rounds % 50 == 0):
                print(".", end="", flush=True)
            if(num_rounds == MAXIMUM_ROUNDS):
                raise Exception("Reached the maXimum number of generation rounds. Probably, you're trying to place too many blocks (>25)")
            block_positions = []
            num_generated = 0
            free_grid_pool = set(TRAIN_FREE_GRID)
            free_grid_pool.remove(agent_position)
            num_rounds += 1

        position = random.sample(free_grid_pool, 1)[0]
        block_positions.append(position)
        if(_all_reachable(set(block_positions), agent_position)):
            num_generated += 1
        else:
            block_positions.pop()

        free_grid_pool.remove(position)
    print("\nSucceeded (Num rounds = " + str(num_rounds + 1) + ")")

    # Draw blocks at sampled positions
    for position, block in zip(block_positions, blocks_to_generate):
        _draw_block(drawing_decorator, room_start[0] + position[0], 1, room_start[1] + position[1], "wool", block)

    mission = MalmoPython.MissionSpec(ET.tostring(tree.getroot()), True)
    # Agent must be placed in the center of the grid cell
    mission.startAt(room_start[0] + agent_position[0] + 0.5, 1, room_start[1] + agent_position[1] + 0.5)

    return mission

def _all_reachable(block_positions, agent_position):
    grid_positions = set(TRAIN_FREE_GRID)
    already_seen = set()
    num_seen_blocks = 0
    stack = [agent_position]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while(num_seen_blocks != len(block_positions) and len(stack) != 0):
        cur_node = stack.pop()
        if(cur_node in already_seen):
            continue
        # Find all unseen neighbors and check whether they're target blocks or not
        for dir in directions:
            neighbor_pos = (cur_node[0] + dir[0], cur_node[1] + dir[1])
            if((neighbor_pos in grid_positions) and (neighbor_pos not in already_seen)):
                if(neighbor_pos in block_positions):
                    num_seen_blocks += 1
                    already_seen.add(neighbor_pos)
                else:
                    stack.append(neighbor_pos)

        already_seen.add(cur_node)

    return num_seen_blocks == len(block_positions)
    
def _draw_block(drawing_decorator, x, y, z, block_type, colour):
    attributes = {
        "x": str(x),
        "y": str(y),
        "z": str(z),
        "type": block_type,
        "colour": str.upper(colour)
    }
    ET.SubElement(drawing_decorator, "ns0:DrawBlock", attributes)
    