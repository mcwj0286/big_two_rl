from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Big2DecisionTree:
    def __init__(self, max_depth=10):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        
    def train(self, states, actions):
        """Train the decision tree on game states and corresponding actions"""
        self.model.fit(states, actions)
        
    def predict(self, state, available_actions):
        """Predict action given game state and mask with available actions"""
        # Get raw predictions
        action_probs = self.model.predict_proba(state.reshape(1, -1))[0]
        
        # Mask unavailable actions
        masked_probs = action_probs + available_actions
        
        # Select highest probability valid action
        if np.sum(masked_probs) == 0:
            # If no valid actions with probability, choose random available action
            valid_indices = np.where(available_actions == 1)[0]
            return np.random.choice(valid_indices)
        
        return np.argmax(masked_probs)
    
    def save(self, filepath):
        """Save the model to a file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath):
        """Load the model from a file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
