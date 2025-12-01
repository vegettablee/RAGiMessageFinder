import torch
import os
import random
import numpy as np
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, filepath='checkpoints/checkpoint.pt'):
    """
    Save a training checkpoint with model state, optimizer state, and training info.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        # Save random states for reproducibility
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath='checkpoints/checkpoint.pt'):
    """
    Load a training checkpoint and restore model and optimizer states.

    Args:
        model: The model to load weights into
        optimizer: The optimizer to restore state
        filepath: Path to the checkpoint file

    Returns:
        tuple: (epoch, loss) - The epoch and loss from the checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file {filepath} not found")
        return 0, None

    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Restore random states for reproducibility
    if 'random_state' in checkpoint:
        random.setstate(checkpoint['random_state'])
    if 'numpy_random_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_random_state'])
    if 'torch_random_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_random_state'])

    print(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
    return epoch, loss


def save_model(model, filepath='saved_models/model.pt'):
    """
    Save only the model state dict (for final model or inference).

    Args:
        model: The model to save
        filepath: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath='saved_models/model.pt'):
    """
    Load model weights from a state dict file.

    Args:
        model: The model to load weights into
        filepath: Path to the model file

    Returns:
        model: The model with loaded weights
    """
    if not os.path.exists(filepath):
        print(f"Model file {filepath} not found")
        return model

    model.load_state_dict(torch.load(filepath, weights_only=False))
    print(f"Model loaded from {filepath}")
    return model


def save_training_history(history, filepath='training_history.pt'):
    """
    Save training history (losses, metrics, etc.).

    Args:
        history: Dictionary containing training history
                 e.g., {'train_loss': [...], 'val_loss': [...], 'epochs': [...]}
        filepath: Path to save the history
    """
    torch.save(history, filepath)
    print(f"Training history saved to {filepath}")


def load_training_history(filepath='training_history.pt'):
    """
    Load training history.

    Args:
        filepath: Path to the history file

    Returns:
        dict: Training history dictionary
    """
    if not os.path.exists(filepath):
        print(f"History file {filepath} not found")
        return {}

    history = torch.load(filepath, weights_only=False)
    print(f"Training history loaded from {filepath}")
    return history


def save_best_model(model, current_loss, best_loss, filepath='saved_models/best_model.pt'):
    """
    Save model if current loss is better than best loss.

    Args:
        model: The model to save
        current_loss: Current validation loss
        best_loss: Best validation loss so far
        filepath: Path to save the best model

    Returns:
        float: Updated best loss
    """
    if best_loss is None or current_loss < best_loss:
        save_model(model, filepath)
        print(f"New best model saved! Loss: {current_loss:.4f}")
        return current_loss
    return best_loss
