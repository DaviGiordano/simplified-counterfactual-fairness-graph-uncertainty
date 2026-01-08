"""
Configuration loader for hyperparameter tuning.
Loads and validates YAML configurations for classifiers and tuning parameters.
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ClassifierConfigLoader:
    """Load and validate classifier configurations from YAML files."""
    
    def __init__(self, config_dir: Path):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing classifier configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        
    def load_classifier_config(self, classifier_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific classifier.
        
        Args:
            classifier_name: Name of the classifier (LR, RF, GB, FAIRGBM)
            
        Returns:
            Dictionary containing classifier configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        if classifier_name in self._configs:
            return self._configs[classifier_name]
            
        # Map classifier names to config files
        config_mapping = {
            "LR": "logistic_regression.yaml",
            "RF": "random_forest.yaml", 
            "GB": "gradient_boosting.yaml",
            "FAIRGBM": "fairgbm.yaml"
        }
        
        if classifier_name not in config_mapping:
            raise ValueError(f"Unknown classifier: {classifier_name}")
            
        config_file = self.config_dir / config_mapping[classifier_name]
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate configuration
            self._validate_config(config, classifier_name)
            
            self._configs[classifier_name] = config
            logger.info(f"Loaded configuration for {classifier_name}")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
            
    def _validate_config(self, config: Dict[str, Any], classifier_name: str) -> None:
        """
        Validate classifier configuration.
        
        Args:
            config: Configuration dictionary
            classifier_name: Name of the classifier
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        # Get the classifier config (first key in the dict)
        classifier_config = list(config.values())[0]
        
        required_fields = ["classpath", "optimization_objective", "search_space"]
        for field in required_fields:
            if field not in classifier_config:
                raise ValueError(f"Missing required field '{field}' in {classifier_name} config")
                
        # Validate search space
        search_space = classifier_config["search_space"]
        if not isinstance(search_space, dict):
            raise ValueError("search_space must be a dictionary")
            
        # Validate each parameter in search space
        for param_name, param_config in search_space.items():
            self._validate_parameter_config(param_name, param_config)
            
    def _validate_parameter_config(self, param_name: str, param_config: Any) -> None:
        """
        Validate individual parameter configuration.
        
        Args:
            param_name: Name of the parameter
            param_config: Parameter configuration
            
        Raises:
            ValueError: If parameter configuration is invalid
        """
        if isinstance(param_config, list):
            # Categorical parameter
            if not param_config:
                raise ValueError(f"Empty categorical values for parameter {param_name}")
            return
            
        if not isinstance(param_config, dict):
            raise ValueError(f"Parameter {param_name} config must be dict or list")
            
        # Numeric parameter
        required_fields = ["type", "range"]
        for field in required_fields:
            if field not in param_config:
                raise ValueError(f"Missing required field '{field}' for parameter {param_name}")
                
        if param_config["type"] not in ["int", "float"]:
            raise ValueError(f"Invalid type for parameter {param_name}: {param_config['type']}")
            
        if not isinstance(param_config["range"], list) or len(param_config["range"]) != 2:
            raise ValueError(f"Invalid range for parameter {param_name}: {param_config['range']}")
            
        # Validate range values
        min_val, max_val = param_config["range"]
        if min_val >= max_val:
            raise ValueError(f"Invalid range for parameter {param_name}: min >= max")
            
    def get_search_space(self, classifier_name: str) -> Dict[str, Any]:
        """
        Get the search space for a classifier.
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            Dictionary containing the search space
        """
        config = self.load_classifier_config(classifier_name)
        classifier_config = list(config.values())[0]
        return classifier_config["search_space"]
        
    def get_optimization_objective(self, classifier_name: str) -> Union[str, Dict[str, Any]]:
        """
        Get the optimization objective for a classifier.
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            Optimization objective (string for single-objective, dict for multi-objective)
        """
        config = self.load_classifier_config(classifier_name)
        classifier_config = list(config.values())[0]
        return classifier_config["optimization_objective"]
        
    def get_default_params(self, classifier_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a classifier.
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            Dictionary containing default parameters
        """
        config = self.load_classifier_config(classifier_name)
        classifier_config = list(config.values())[0]
        return classifier_config.get("default_params", {})
        
    def get_classpath(self, classifier_name: str) -> str:
        """
        Get the classpath for a classifier.
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            Classpath string
        """
        config = self.load_classifier_config(classifier_name)
        classifier_config = list(config.values())[0]
        return classifier_config["classpath"]


class TuningConfigLoader:
    """Load and validate tuning configuration."""
    
    def __init__(self, config_file: Path):
        """
        Initialize the tuning configuration loader.
        
        Args:
            config_file: Path to the tuning configuration file
        """
        self.config_file = Path(config_file)
        self._config = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load tuning configuration from file.
        
        Returns:
            Dictionary containing tuning configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        if self._config is not None:
            return self._config
            
        if not self.config_file.exists():
            raise FileNotFoundError(f"Tuning configuration file not found: {self.config_file}")
            
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self._validate_config(config)
            self._config = config
            
            logger.info(f"Loaded tuning configuration from {self.config_file}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_file}: {e}")
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate tuning configuration.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        # Validate tuning section
        if "tuning" not in config:
            raise ValueError("Missing 'tuning' section in configuration")
            
        tuning_config = config["tuning"]
        required_fields = ["optimizer", "n_trials", "cv_folds", "random_state"]
        for field in required_fields:
            if field not in tuning_config:
                raise ValueError(f"Missing required field '{field}' in tuning config")
                
        # Validate optimizer
        if tuning_config["optimizer"] not in ["optuna", "grid", "random"]:
            raise ValueError(f"Invalid optimizer: {tuning_config['optimizer']}")
            
        # Validate numeric fields
        if not isinstance(tuning_config["n_trials"], int) or tuning_config["n_trials"] <= 0:
            raise ValueError("n_trials must be a positive integer")
            
        if not isinstance(tuning_config["cv_folds"], int) or tuning_config["cv_folds"] <= 0:
            raise ValueError("cv_folds must be a positive integer")
            
        if not isinstance(tuning_config["random_state"], int):
            raise ValueError("random_state must be an integer")
            
    def get_tuning_config(self) -> Dict[str, Any]:
        """Get the tuning configuration section."""
        config = self.load_config()
        return config["tuning"]
        
    def get_objectives_config(self) -> Dict[str, Any]:
        """Get the objectives configuration section."""
        config = self.load_config()
        return config.get("objectives", {})
        
    def get_pruning_config(self) -> Dict[str, Any]:
        """Get the pruning configuration section."""
        config = self.load_config()
        return config.get("pruning", {})
        
    def get_parallel_config(self) -> Dict[str, Any]:
        """Get the parallel configuration section."""
        config = self.load_config()
        return config.get("parallel", {})
        
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get the MLflow configuration section."""
        config = self.load_config()
        return config.get("mlflow", {})

