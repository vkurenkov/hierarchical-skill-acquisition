# hierarchical-skill-acquisition
Implementation of the Hierarchical and Interpretable Skill Acquisition in Multi-task Reinforcement Learning by Tianmin Shu, Caiming Xiong, and Richard Socher 

## Paper Overview
  
The paper proposes a new method for solving multi-task environments [1]. Authors introduce another hierarchical approach and compare it to other methods such as ["Flat" policy](https://arxiv.org/abs/1706.06551) and [H-DRLN](https://arxiv.org/abs/1604.07255).


#### Key ideas:
  * Hierarchical Design
  * Interpretable Policies
  * Curriculum Learning
  
#### Architecture
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/4092658/41410751-1db9aac8-6fe3-11e8-8589-dc526656e91e.png" alt="Architecture" width="60%" height="60%">
</p>
  
The picture above represents the proposed architecture. This architecture can be summarized in one sentence: At any given moment of time _t_, it decides whether to use one of the already trained policies for a chosen sub-task or to act on its own (low-level actions).

All the way down to LSTM, we encode current state. Then we must decide on several things:
  - What sub-task policy should we use? (Instruction Policy, here comes the interpretability property)
  - Should we use a chosen sub-task policy? (Switch Policy)
  - If we do not use the sub-task policy, what should we do? (Augmented Policy)
  
If we decided to switch to the sub-task policy, we use Base Policy module. It represents the same architecture described above thus we can go deeper and deeper, infinity and beyond.

The policy is optimized using Advantage-Actor Critic, why not the A3C? - Authors left it as a possible future work.

#### Training Process

To make this architecture work, we need to manually specify the order of the tasks and pre-train the policy at the zero-level. Particularly, authors work with this curriculum: "Find object" -> "Get object" -> "Put object" -> "Stack object". 

"Find object" is the zero-level policy hence it must be pre-trained before moving to the next level task ("Get object").

#### Information Reference:
  1. Multi-task environment - an environment where the main goal of the agent is to find a trajectory to solve a problem that consists of another smaller problems, e.g. to solve the instruction "Get object", the agent must be able to solve "Find object".
 
## Milestones:

**1.** Set up the environment
- [X] Define the training environment
- [X] Define the testing environment
- [X] Implement blocks/agent random placement for the training environment
- [ ] Implement blocks/agent random placement for the testing environment
- [ ] Generate tasks

**2.** Build RL models
- [ ] Implement "flat" model
- [ ] Implement hierarchical model
  - [X] "Flat" part
  - [X] Augmented policy
  - [X] Switch policy
  - [X] Instruction policy
  - [ ] Use of base policy
  - [ ] A2C optimization
  - [ ] Stochastic Temporal Grammar
  
**3.** Train the agent
- [ ] Flat policy
  - [ ] Task #1 - Find x
  - [ ] Task #2 - Get x
  - [ ] Task #3 - Put x
  - [ ] Task #4 - Stack x
- [ ] Hierarchical policy
  - [ ] Task #1 - Find x
  - [ ] Task #2 - Get x
  - [ ] Task #3 - Put x
  - [ ] Task #4 - Stack x
