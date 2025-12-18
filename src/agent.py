import random
from typing import Dict, Tuple

# State definition matching Env: (car_lane, dL, dM, dR)
State = tuple[int, int, int, int]

class QLearningAgent:
    def __init__(self, alpha=0.20, gamma=0.95, epsilon=1.0, epsilon_decay=0.990, seed=7):
        """
        Inicializa el agente Q-Learning.
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        :param epsilon: Probabilidad de exploración inicial.
        :param epsilon_decay: Factor de decaimiento de epsilon.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        
        # Q-table: Map state -> [q_left, q_stay, q_right]
        self.q_table: Dict[State, list[float]] = {}
        
    def get_q(self, state: State):
        """Devuelve los valores Q para un estado dado, inicializándolos si no existen."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def act(self, state: State, training: bool = True):
        """
        Elige una acción basada en el estado actual.
        Si training=True, usa epsilon-greedy.
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, 2)
        
        q_vals = self.get_q(state)
        # Argmax with tie breaking
        max_v = max(q_vals)
        candidates = [i for i, v in enumerate(q_vals) if v == max_v]
        return self.rng.choice(candidates)

    def learn(self, s: State, a: int, r: float, s2: State, done: bool):
        """Actualiza la tabla Q usando la ecuación de Bellman."""
        q_old = self.get_q(s)[a]
        
        if done:
            target = r
        else:
            target = r + self.gamma * max(self.get_q(s2))
            
        # Update
        self.q_table[s][a] += self.alpha * (target - q_old)

    def decay(self):
        """Reduce el valor de epsilon para disminuir la exploración con el tiempo."""
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
