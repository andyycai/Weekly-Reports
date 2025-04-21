#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 9 18:00:00 2025

@author: Andy.
"""

import mesa
import networkx as nx
from enum import Enum


class State(Enum):
    """Define the possible states for agents:"""
    
    NONSMOKER = 0
    SMOKER = 1
    QUITTER = 2


class NSQ_Model(mesa.Model):
    """Define the model-level actions as NSQ_Model:"""
    
    def __init__(
            self,
            # Set seed:
            # seed = 
            # Embedded network structure: (not specified now, specify when implementing)
            network_type
            # Spontaneous parameters (3):
            delta_N_to_S = ,
            delta_S_to_Q = ,
            delta_Q_to_S = ,
            # Interaction parameters (4):
            beta_N_to_S_due_to_S = ,
            beta_S_to_Q_due_to_N = ,
            beta_S_to_Q_due_to_Q = ,
            beta_Q_to_S_due_to_S = ,
            # Initial population parameters (%, SMOKER, QUITTER):
            initial_SMOKER_pct = ,
            initial_QUITTER_pct = ,
            # Embedded network's parameters: (not specified now, specify when implementing)
            **network_parameters
            ):
        """Initialise the NSQ_Model class:"""
        
        # Initialise Mesa's 'Model' class:
        super().__init__() # super().__init__(seed) if seed != None
        
        # Pass the values of spontaneous parameters to the model class which is under construction:
        self.delta_N_to_S = delta_N_to_S
        self.delta_S_to_Q = delta_S_to_Q
        self.delta_Q_to_S = delta_Q_to_S
        
        # Pass the values of interaction parameters to the model class which is under construction:
        self.beta_N_to_S_due_to_S = beta_N_to_S_due_to_S
        self.beta_S_to_Q_due_to_N = beta_S_to_Q_due_to_N
        self.beta_S_to_Q_due_to_Q = beta_S_to_Q_due_to_Q
        self.beta_Q_to_S_due_to_S = beta_Q_to_S_due_to_S
        
        # Define estimated interaction parameters (from Christakis):
        self.beta_N_to_S_due_to_S_est = 1 - (1 - beta_N_to_S_due_to_S * (1 - (1 - delta_N_to_S)**4.5)) ** (1/4.5)
        self.beta_S_to_Q_due_to_Q_est = 1 - (1 - beta_S_to_Q_due_to_Q * (1 - (1 - delta_S_to_Q)**4.5)) ** (1/4.5)
        
        # Pass the values of initial population parameters to the model class which is under construction:
        self.initial_NONSMOKER_pct = 1 - (initial_SMOKER_pct + initial_QUITTER_pct)
        self.initial_SMOKER_pct = initial_SMOKER_pct
        self.initial_QUITTER_pct = initial_QUITTER_pct
        
        # Pass the embedded network's details to the model class which is under construction:
        self.graph = network_type(**network_parameters)
        self.grid = mesa.space.NetworkGrid(self.graph)
        
        
        # [Question: put where? | Answer: put in the step function]
        # [!!!] Make sure all agents are updated at the same time, based on their states before the current time step:
        # self.schedule = SimultaneousActivation(self)
        # Mesa 3 migration: self.agents.do("step")
        #                   self.agents.do("advance")
        
        
        
        # Adding agents to nodes:
        for node in self.graph.nodes(): # [!] self.graph.nodes(): an iterable NodeView, not a list
            agent = NSQ_Agent(self, State.NONSMOKER) # Mesa 3 migration: no need to pass unique_id explicitly
            # self.schedule.add(agent) Mesa 3 migration: eliminating the need to manually call ~
            self.grid.place_agent(agent, node) # Mesa automatically assign agent.pos = node
            # when calling agent class, the objects in agent class are automatically passed
            # (agent, node) = (i+1, i), i starts from 0
            # [!] implicitly guarantee # of agents and # of nodes are equal
        
        # Randomly choose initial SMOKER from all agents (choose nodes first):
        initial_SMOKER_nodes = self.random.sample(list(self.graph.nodes()), round(self.initial_SMOKER_pct * len(list(self.graph.nodes()))))
        initial_SMOKER_agents = self.grid.get_cell_list_contents(initial_SMOKER_nodes)
        for agent in initial_SMOKER_agents:
            agent.current_state = State.SMOKER
            agent.updated_state = State.SMOKER
        
        # Randomly choose initial QUITTER from all remaining agents (choose nodes first):
        initial_QUITTER_nodes = self.random.sample(
            list(filter(lambda node: node not in initial_SMOKER_nodes, list(self.graph.nodes()))),
            round(self.initial_QUITTER_pct * len(list(self.graph.nodes())))
            )
        initial_QUITTER_agents = self.grid.get_cell_list_contents(initial_QUITTER_nodes)
        for agent in initial_QUITTER_agents:
            agent.current_state = State.QUITTER
            agent.updated_state = State.QUITTER
        
        # Generate initial NONSMOKER by deleting initial SMOKER and initial QUITTER:
        initial_NONSMOKER_nodes = list(filter(lambda node: node not in initial_SMOKER_nodes and node not in initial_QUITTER_nodes, list(self.graph.nodes())))
        initial_NONSMOKER_agents = self.grid.get_cell_list_contents(initial_NONSMOKER_nodes)
        
        
        # [Question: choose which version? | Answer: Version 1]
        # Add DataCollector to track desired data:
        
        # Version 1:
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # Agents' list in each state:
                "NONSMOKER_nodes": lambda m: m.get_agents_by_state(State.NONSMOKER),
                "SMOKER_nodes": lambda m: m.get_agents_by_state(State.SMOKER),
                "QUITTER_nodes": lambda m: m.get_agents_by_state(State.QUITTER),
                
                # Node lists of agents in each state:
                "NONSMOKER_nodes": lambda m: m.get_nodes_by_state(State.NONSMOKER),
                "SMOKER_nodes": lambda m: m.get_nodes_by_state(State.SMOKER),
                "QUITTER_nodes": lambda m: m.get_nodes_by_state(State.QUITTER),
        
                # Counts of agents in each state:
                "NONSMOKER_count": lambda m: m.count_agents_by_state(State.NONSMOKER),
                "SMOKER_count": lambda m: m.count_agents_by_state(State.SMOKER),
                "QUITTER_count": lambda m: m.count_agents_by_state(State.QUITTER),
                
                # Percentages of agents in each state:
                "% NONSMOKER": lambda m: m.pct_agents_by_state(State.NONSMOKER),
                "% SMOKER": lambda m: m.pct_agents_by_state(State.SMOKER),
                "% QUITTER": lambda m: m.pct_agents_by_state(State.QUITTER)
            },
            agent_reporters={
                "State": lambda a: a.current_state, # "current_state.name"
                "Node": lambda a: a.pos, # "pos"
            }
        )
        
        # Version 2:
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # Agents' list in each state:
                "NONSMOKER_nodes": self.get_agents_by_state(State.NONSMOKER),
                "SMOKER_nodes": self.get_agents_by_state(State.SMOKER),
                "QUITTER_nodes": self.get_agents_by_state(State.QUITTER),
                
                # Node lists of agents in each state:
                "NONSMOKER_nodes": self.get_nodes_by_state(State.NONSMOKER),
                "SMOKER_nodes": self.get_nodes_by_state(State.SMOKER),
                "QUITTER_nodes": self.get_nodes_by_state(State.QUITTER),
                
                # Counts of agents in each state:
                "NONSMOKER_count": self.count_agents_by_state(State.NONSMOKER),
                "SMOKER_count": self.count_agents_by_state(State.SMOKER),
                "QUITTER_count": self.count_agents_by_state(State.QUITTER),
                
                # Percentages of agents in each state:
                "% NONSMOKER": self.pct_agents_by_state(State.NONSMOKER),
                "% SMOKER": self.pct_agents_by_state(State.SMOKER),
                "% QUITTER": self.pct_agents_by_state(State.QUITTER)
            },
            agent_reporters={
                "State": self.current_state, # "current_state.name"
                "Node": self.pos # "pos"
            }
        )
        
        
        # Capture desired data from initial settings (step 0):
        self.datacollector.collect(self)
    
    
    def get_agents_by_state(self, state):
        """Return list of agents in a given state:"""
        return list(filter(lambda agent: agent.current_state is state, self.agents))
    
    def get_nodes_by_state(self, state):
        """Return list of nodes where their agents in a given state:"""
        return list(map(lambda agent: agent.pos, self.get_agents_by_state(state)))
    
    def count_agents_by_state(self, state):
        """Calculate counts of agents in a given state:"""
        return len(self.get_agents_by_state(state))

    def pct_agents_by_state(self, state):
        """Calculate percentage of agents in a given state:"""
        return (self.count_agents_by_state(state) / len(self.agents)) * 100
    
    def step(self):
        """Advance model by one step:"""
        self.agents.do("step")
        self.agents.do("advance")
        self.datacollector.collect(self)
        
    def run_model(self, n):
        """Run model for desired number of iterations"""
        for _ in range(n):
            self.step()
        
    def get_pct_data(self):
        """Returns percentage data by state:"""
        data = self.datacollector.get_model_vars_dataframe()
        return data[["% NONSMOKER", "% SMOKER", "% QUITTER"]]
    
    def print_pct_data(self):
        """Print percentage data by state:"""
        print(self.get_pct_data().to_string(
            float_format="%.1f",
            )
        )


class NSQ_Agent(mesa.Agent):
    """Define the agent-level actions as NSQ_Agent:"""
    
    def __init__(
            self,
            model,
            initial_state
            ):
        """Initialise the NSQ_Agent class:"""
        
        # Initialise Mesa's 'Agent' class and pass objects from the 'model' class to the agent class which is under construction:
        super().__init__(model)

        # Define current state before state change and initialise:
        self.current_state = initial_state
        
        # Define updated state after state change and initialise:
        self.updated_state = initial_state
        
    def NONSMOKER_initiation(self):
        """Define the dynamics of smoking initiation for NONSMOKER:"""
        
        # Identify neighbouring agents and those who are SMOKER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_N_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S)**len(SMOKER_neighbours))
            # prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S_est)**len(SMOKER_neighbours))
            if self.random.random() < prob_N_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours
    
    def SMOKER_cessation(self):
        """Define the dynamics of smoking cessation for SMOKER:"""
        
        # Identify neighbouring agents and those who are NONSMOKER and QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        NONSMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.NONSMOKER, neighbours))
        # NONSMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.NONSMOKER)]
        QUITTER_neighbours = list(filter(lambda agent: agent.current_state is State.QUITTER, neighbours))
        # QUITTER_neighbours = [agent for agent in neighbours if (agent.current_state is State.QUITTER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_S_to_Q:
            self.updated_state = State.QUITTER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_S_to_Q_due_to_N = (len(NONSMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_N)**len(NONSMOKER_neighbours))
            prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q)**len(QUITTER_neighbours))
            # prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q_est)**len(QUITTER_neighbours))
            if self.random.random() < prob_S_to_Q_due_to_N:
                self.updated_state = State.QUITTER
            elif self.random.random() < prob_S_to_Q_due_to_Q:
                self.updated_state = State.QUITTER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, NONSMOKER_neighbours, QUITTER_neighbours
    
    def QUITTER_relapse(self):
        """Define the dynamics of smoking relapse for QUITTER:"""
        
        # Identify neighbouring agents and those who are QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_Q_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_Q_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_Q_to_S_due_to_S)**len(SMOKER_neighbours))
            if self.random.random() < prob_Q_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours

    def step(self):
        """Decide what updated state should be for one iteration:"""
        
        # Smoking initiation for NONSMOKER:
        if self.current_state is State.NONSMOKER:
            self.NONSMOKER_initiation()
        
        # Smoking cessation for SMOKER:
        elif self.current_state is State.SMOKER:
            self.SMOKER_cessation()
        
        # Smoking relapse for QUITTER:
        elif self.current_state is State.QUITTER:
            self.QUITTER_relapse()
    
    def advance(self):
        """Apply any state change decided in step() to the agent's current state:"""
        self.current_state = self.updated_state
    
    # NOTE: The purpose of advance():
    #     - apply the decision-making of any state change to agent
    #     - separate decision-making phase and state-application phase to avoid one agent's
    #       decision being influenced by another one's
    # => - advanced() is automatically called by SimultaneousActivation scheduler
    #    - SimultaneousActivation scheduler automatically handles the sequence of calling
    #      step() for each agent and then calling advance() for each agent afterward

        
###############################################################################


"""Spontaneity-based State Change"""

"""NONSMOKER ==> SMOKER:"""
# - delta_N_to_S

"""SMOKER ==> QUITTER:"""
# - delta_S_to_Q

"""QUIITER ==> SMOKER:"""
# - delta_Q_to_S


"""Binomial Approximation for Interaction-based State Change"""

"""NONSMOKER ==> SMOKER (due to interactions with SMOKER neighbours):"""
# - beta_N_to_S_due_to_S is the probability of successful interaction of an NONSMOKER agent with a SMOKER agent.
# => Probability of an NONSMOKER agent performs a state change to a SMOKER agent due to interactions with SMOKER neighbours:
# prob_N_to_S_due_to_S = (#_SMOKER_neighbours / #_neighbours) * (1 - (1-beta_N_to_S_due_to_S)**(#_SMOKER_neighbours))

"""SMOKER ==> QUITTER (due to interactions with NONSMOKER neighbours):"""
# - beta_S_to_Q_due_to_N is the probability of successful interaction of a SMOKER agent with an NONSMOKER agent.
# => Probability of a SMOKER agent performs a state change to a QUITTER agent due to interactions with NONSMOKER neighbours:
# prob_S_to_Q_due_to_N = (#_NONSMOKER_neighbours / #_neighbours) * (1 - (1-beta_S_to_Q_due_to_N)**(#_NONSMOKER_neighbours))

"""SMOKER ==> QUITTER (due to interactions with QUITTER neighbours):"""
# - beta_S_to_Q_due_to_Q is the probability of successful interaction of a SMOKER agent with a QUITTER agent.
# => Probability of a SMOKER agent performs a state change to a QUITTER agent due to interactions with QUITTER neighbours:
# prob_S_to_Q_due_to_Q = (#_QUITTER_neighbours / #_neighbours) * (1 - (1-beta_S_to_Q_due_to_Q)**(#_QUITTER_neighbours))

"""QUIITER ==> SMOKER (due to interactions with SMOKER neighbours):"""
# - beta_Q_to_S_due_to_S is the probability of successful interaction of a QUITTER agent with a SMOKER agent.
# => Probability of a QUITTER agent performs a state change to a SMOKER agent due to interactions with SMOKER neighbours:
# prob_Q_to_S_due_to_S = (#_SMOKER_neighbours / #_neighbours) * (1 - (1-beta_Q_to_S_due_to_S)**(#_SMOKER_neighbours))

# NOTE: pow(a,b) is equivalent to a**b.


"""Estimation of Interaction Parameters (from Christakis)"""

"""NONSMOKER ==> SMOKER (due to interactions with SMOKER neighbours):"""
# - beta_N_to_S_due_to_S_est = 1 - (1 - beta_N_to_S_due_to_S * (1 - (1 - delta_N_to_S)**4.5)) ** (1/4.5)

"""SMOKER ==> QUITTER (due to interactions with QUITTER neighbours):"""
# - beta_S_to_Q_due_to_Q_est = 1 - (1 - beta_S_to_Q_due_to_Q * (1 - (1 - delta_S_to_Q)**4.5)) ** (1/4.5)


###############################################################################

