import torch
import numpy as np
import os
import wandb
from typing import Dict, Optional
from enum import Enum
from manabot.model.agent import Agent
from collections import Counter

from manabot.env import ObservationSpace
from manabot.infra.log import getLogger
from manabot.infra.hypers import ObservationSpaceHypers, AgentHypers
# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

    
def load_model_from_wandb(
    model: str,
    version: str = "latest", 
    project: Optional[str] = None,
    device: str = "cpu"
) -> Agent:
    """
    Load a trained model from wandb artifacts with minimal wandb interaction.
    
    Args:
        model: Name of the model (e.g. "quick_train")
        version: Version string (e.g. "v3" or "latest")
        project: Wandb project name
        device: Device to load model on ("cpu" or "cuda")
        
    Returns:
        Loaded agent model ready for inference
    """
    logger = getLogger(__name__).getChild("load_model_from_wandb")
    try:
        # Force online mode to ensure artifact can be fetched
        os.environ['WANDB_MODE'] = 'online'
        
        # Use a silent API object without starting a run
        api = wandb.Api()
        artifact_path = f"{project or 'manabot'}/{model}:{version}"
        logger.info(f"Loading artifact: {artifact_path}")
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download("/tmp")
        
        # Directly use the expected filename pattern based on the save method
        potential_paths = [
            os.path.join(artifact_dir, f"{model}.pt"),
            os.path.join(artifact_dir, f"{model}_latest.pt"),
        ]
        
        # Also look for any .pt files
        pt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        potential_paths.extend([os.path.join(artifact_dir, f) for f in pt_files])
        
        # Find the first valid checkpoint file
        checkpoint_path = None
        for path in potential_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError(f"No .pt files found in the artifact at {artifact_dir}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Debug: print the checkpoint keys
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Look for model state dict with flexible key names
        state_dict_key = None
        for key in ['agent_state_dict', 'model_state_dict', 'state_dict']:
            if key in checkpoint:
                state_dict_key = key
                break
                
        if state_dict_key is None:
            raise ValueError(f"Checkpoint does not contain a state dictionary. Keys found: {list(checkpoint.keys())}")
            
        logger.info(f"Using state dictionary from key: {state_dict_key}")
        
        assert 'hypers' in checkpoint
        logger.info("Found hyperparameters in checkpoint")
        hypers_dict = checkpoint['hypers']
            
        assert 'observation_hypers' in hypers_dict
        obs_hypers = ObservationSpaceHypers(**hypers_dict['observation_hypers'])
        logger.info(f"Using saved observation hyperparameters")
        
        assert 'agent_hypers' in hypers_dict
        agent_hypers = AgentHypers(**hypers_dict['agent_hypers'])
        logger.info(f"Using saved agent hyperparameters (attention_on={agent_hypers.attention_on})")
        
        # Create observation space and agent with the appropriate hyperparameters
        obs_space = ObservationSpace(obs_hypers)
        agent = Agent(obs_space, agent_hypers)
        logger.info(f"Created model with {'saved' if 'hypers' in checkpoint or 'agent_hypers' in checkpoint else 'default'} hyperparameters")
        
        # Load model weights
        agent.load_state_dict(checkpoint[state_dict_key])
        agent.eval()
        agent = agent.to(device)
        
        # See if we have information about training steps
        if 'global_step' in checkpoint:
            logger.info(f"Model was trained for {checkpoint['global_step']} steps")
        
        logger.info(f"Successfully loaded model")
        return agent
    except Exception as e:
        logger.error(f"Error loading model from wandb: {e}")
        # If the exception relates to artifact not found, show clear message
        if "not found" in str(e).lower():
            logger.error(f"Could not find artifact '{model}:{version}'. Check if the name is correct and the artifact exists.")
        # If the exception relates to model loading, print more details
        elif "state dictionary" in str(e) or "state_dict" in str(e):
            logger.error("The model file was found but its structure doesn't match expectations.")
            logger.error("This could happen if the model was saved with a different format or version.")
        raise


# -----------------------------------------------------------------------------
# Player Classes
# -----------------------------------------------------------------------------

class PlayerType(Enum):
    """Types of players for evaluation."""
    MODEL = "model"
    RANDOM = "random"
    DEFAULT = "default"

class Player:
    """Base player class for evaluation."""
    def __init__(self, name: str, player_type: PlayerType):
        self.name = name
        self.player_type = player_type
        self.device = "cpu"
        self.wins = 0
        self.games = 0
        self.action_history = []  # Track actions for analysis
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from observation."""
        raise NotImplementedError
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.wins / self.games if self.games > 0 else 0.0
    
    def record_result(self, won: bool) -> None:
        """Record game result."""
        self.games += 1
        if won:
            self.wins += 1
    
    def record_action(self, action: int, obs: Dict[str, np.ndarray]) -> None:
        """Record an action for later analysis."""
        self.action_history.append((action, obs.get("actions", []).shape[0]))
    
    def get_action_distribution(self) -> Dict[str, float]:
        """Get distribution of action types."""
        if not self.action_history:
            return {}
        
        action_counts = Counter([action for action, _ in self.action_history])
        total = len(self.action_history)
        return {f"action_{action}": count/total for action, count in action_counts.items()}
    
    def reset_history(self) -> None:
        """Reset action history."""
        self.action_history = []
    
    def to(self, device: str) -> 'Player':
        """Move player to specified device."""
        self.device = device
        return self

class ModelPlayer(Player):
    """Player that uses a trained model for inference."""
    def __init__(
        self, 
        name: str, 
        agent: Agent, 
        deterministic: bool = False,
        record_logits: bool = False
    ):
        super().__init__(name, PlayerType.MODEL)
        self.agent = agent
        self.deterministic = deterministic
        self.device = next(agent.parameters()).device
        self.record_logits = record_logits
        self.logits_history = [] if record_logits else None
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from model."""
        # Convert numpy arrays to tensors with batch dimension
        tensor_obs = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }
        
        # Get action from model
        with torch.no_grad():
            logits, _ = self.agent(tensor_obs)
            if self.record_logits:
                assert self.logits_history is not None
                self.logits_history.append(logits.detach().cpu().numpy())
                
            action, _, _, _ = self.agent.get_action_and_value(
                tensor_obs, deterministic=self.deterministic)
            
            # Record the action for analysis
            action_value = action.item()
            self.record_action(action_value, obs)
            return action_value
    
    def get_action_confidence(self) -> Dict[str, float]:
        """Get statistics about action confidence."""
        if not self.logits_history:
            return {}
        
        # Calculate softmax probabilities
        probs = []
        for logits in self.logits_history:
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs.append(exp_logits / np.sum(exp_logits, axis=1, keepdims=True))
        
        # Calculate statistics
        chosen_probs = [p.max() for p in probs]
        return {
            "mean_confidence": np.mean(chosen_probs),
            "min_confidence": np.min(chosen_probs),
            "max_confidence": np.max(chosen_probs),
        }
    
    def to(self, device: str) -> 'ModelPlayer':
        """Move player and model to device."""
        super().to(device)
        self.agent = self.agent.to(device)
        return self

class RandomPlayer(Player):
    """Player that selects random valid actions."""
    def __init__(self, name: str):
        super().__init__(name, PlayerType.RANDOM)
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get random valid action."""
        valid_actions = np.where(obs["actions_valid"] > 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        
        action = int(np.random.choice(valid_actions))
        self.record_action(action, obs)
        return action
    
class DefaultPlayer(Player):
    """Player that always plays 0."""
    def __init__(self, name: str):
        super().__init__(name, PlayerType.DEFAULT)
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        return 0