# Introduction

This model simulates how agents move and exchange wealth in a grid.  
The goal is to observe how wealth is distributed across space and among agents after a number of steps.  

# Model Description

- The model is made up of *n* agents placed randomly on a 10x10 grid.  
- Each agent starts with a wealth value randomly chosen between **1 and 5**.  

# Agent Behavior

- At each step (with the total number of steps set to **100**):  
  1. Each agent looks at its neighboring cells.  
  2. If one of them has a higher average wealth per agent than the agent’s own wealth, it moves to that cell.  
  3. Otherwise, the agent stays in its current cell.  
  4. After moving (or staying), the agent randomly gives **1 unit of wealth** to another agent in the same cell.  

# Results

- After **100 steps**, it is possible to observe:  
  - The **spatial distribution of wealth** across the grid.  
  - The **overall distribution of wealth** among all agents.
 
# Graphs: 

Wealth distribution at the beggining of simulation:
<img width="1720" height="352" alt="image" src="https://github.com/user-attachments/assets/ea3cf39b-e34b-47cd-b579-78b0af8ab924" />

Wealth distribution at the end of simulation: 
<img width="1720" height="352" alt="image" src="https://github.com/user-attachments/assets/6cf6aff4-9c5e-4620-99c4-b4984d5fcdb4" />


Grid wealth distribution at the beginning of simulation: (The number inside the grid is the total wealth/n_agents in the grid)

<img width="386" height="342" alt="image" src="https://github.com/user-attachments/assets/e884afc0-9d67-4843-8f3a-ab01b7ee6d1d" />

Grid wealth distribution at the end of simulation: (The number inside the grid is the total wealth/n_agents in the grid)

<img width="370" height="342" alt="image" src="https://github.com/user-attachments/assets/45c9f6ac-23ef-4d6c-ab80-98d5af9a7f4b" />
