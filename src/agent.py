import random
from typing import Dict, Tuple, List, Optional

# State definition matching Env: (car_lane, dL, dM, dR)
# Represents the car's current lane and discretized distances to obstacles in each of the 3 lanes.
State = Tuple[int, int, int, int]

class QLearningAgent:
    """
    A Tabular Q-Learning Agent designed for obstacle avoidance in a 3-lane highway.
    
    The agent maintains a Q-table that maps environment states to expected future rewards
    for each possible action (Move Left, Stay, Move Right). It employs an epsilon-greedy
    strategy to balance exploration of new behaviors with exploitation of learned knowledge.
    
    Attributes:
        alpha (float): Learning rate, determining how much new information overrides old.
        gamma (float): Discount factor, weighting the importance of future rewards.
        epsilon (float): Exploration rate for the epsilon-greedy policy.
        epsilon_decay (float): Multiplicative factor applied to epsilon after each episode.
        q_table (Dict[State, List[float]]): The internal knowledge base mapping states to Q-values.
    """

    def __init__(self, alpha: float = 0.20, gamma: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.990, seed: int = 7):
        """
        Initializes the Q-Learning agent with specified hyperparameters.

        Args:
            alpha (float): Learning rate (0.0 to 1.0). Defaults to 0.20.
            gamma (float): Discount factor for future rewards. Defaults to 0.95.
            epsilon (float): Initial exploration probability. Defaults to 1.0.
            epsilon_decay (float): Rate at which epsilon diminishes. Defaults to 0.990.
            seed (int): Random seed for reproducibility. Defaults to 7.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        
        # Q-table: Map state -> [q_left, q_stay, q_right]
        self.q_table: Dict[State, List[float]] = {}
        
    def get_q(self, state: State) -> List[float]:
        """
        Retrieves Q-values for a given state, initializing them to zero if the state is new.

        Args:
            state (State): The current environment state.

        Returns:
            List[float]: A list of 3 Q-values corresponding to [Left, Stay, Right].
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def act(self, state: State, training: bool = True) -> int:
        """
        Selects an action based on the agent's current policy.

        During training, it uses epsilon-greedy exploration. In evaluation (training=False),
        it strictly follows the greedy policy (highest Q-value).

        Args:
            state (State): The current observation from the environment.
            training (bool): Whether to allow randomized exploration. Defaults to True.

        Returns:
            int: The chosen action (0: Left, 1: Stay, 2: Right).
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, 2)
        
        q_vals = self.get_q(state)
        # Argmax with tie breaking to prevent systematic bias
        max_v = max(q_vals)
        candidates = [i for i, v in enumerate(q_vals) if v == max_v]
        return self.rng.choice(candidates)

    def get_confidence(self, state: State) -> float:
        """
        Calculates the agent's 'confidence' in its current state knowledge.
        
        Confidence is defined here as the margin between the best action's Q-value 
        and the average Q-value for that state. A higher margin indicates a strong 
        preference for a specific maneuver.

        Args:
            state (State): The state to evaluate.

        Returns:
            float: A scalar representing the decision confidence.
        """
        q_vals = self.get_q(state)
        if all(v == 0.0 for v in q_vals):
            return 0.0
        max_v = max(q_vals)
        avg_v = sum(q_vals) / len(q_vals)
        return float(max_v - avg_v)

    def learn(self, s: State, a: int, r: float, s2: State, done: bool):
        """
        Updates the Q-table using the Temporal Difference (TD) error.

        Implements the standard Q-Learning update rule:
        Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        where target is (reward + gamma * max(Q(s', a')))

        Args:
            s (State): The state before taking the action.
            a (int): The action performed.
            r (float): The reward received.
            s2 (State): The state resulting from the action.
            done (bool): Whether the episode ended after this step.
        """
        q_vals = self.get_q(s)
        q_old = q_vals[a]
        
        if done:
            target = r
        else:
            target = r + self.gamma * max(self.get_q(s2))
            
        # Update rule
        self.q_table[s][a] += self.alpha * (target - q_old)

    def decay(self):
        """
        Reduces the exploration rate (epsilon) according to the decay factor.
        Ensures epsilon does not fall below a functional minimum (0.01).
        """
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
