"""
Configuration utilities for Enhanced Polar Registration
"""

import json
import yaml
import os
from typing import Dict, Any
import logging


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif config_path.endswith(('.yml', '.yaml')):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'model',
        'data',
        'loss',
        'training'
    ]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate model configuration
    model_keys = ['in_channels', 'out_channels', 'base_channels']
    for key in model_keys:
        if key not in config['model']:
            logging.error(f"Missing required model configuration key: {key}")
            return False
    
    # Validate data configuration
    data_keys = ['train_txt', 'val_txt', 'template_path', 'target_size']
    for key in data_keys:
        if key not in config['data']:
            logging.error(f"Missing required data configuration key: {key}")
            return False
    
    # Validate training configuration
    training_keys = ['epochs', 'batch_size', 'learning_rate', 'optimizer']
    for key in training_keys:
        if key not in config['training']:
            logging.error(f"Missing required training configuration key: {key}")
            return False
    
    return True


def create_experiment_config(base_config_path: str, 
                           experiment_name: str,
                           overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create experiment configuration from base config with overrides
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of experiment
        overrides: Configuration overrides
    
    Returns:
        Experiment configuration
    """
    # Load base configuration
    config = load_config(base_config_path)
    
    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)
    
    # Set experiment name
    config['training']['experiment_name'] = experiment_name
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    return config
