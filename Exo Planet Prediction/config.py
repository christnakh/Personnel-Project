"""
⚙️ Configuration Management
Centralized configuration for the Exoplanet ML system
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'exoplanet-ml-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Port Configuration
    DEFAULT_PORT = int(os.environ.get('FLASK_PORT', 5001))
    PORT_RANGE = list(range(5001, 5010))  # Try ports 5001-5009
    
    # Model Configuration
    MODELS_DIR = Path("models")
    TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
    DATA_DIR = Path("data")
    LOGS_DIR = Path("logs")
    TEST_RESULTS_DIR = Path("test_results")
    
    # Model Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # API Configuration
    API_TIMEOUT = 30
    MAX_BATCH_SIZE = 1000
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # NASA API Configuration
    NASA_API_TIMEOUT = 30
    NASA_CACHE_DURATION = 3600  # 1 hour
    
    # Performance Configuration
    MAX_PREDICTION_TIME = 5.0  # seconds
    ENABLE_CACHING = True
    
    @classmethod
    def get_available_port(cls):
        """Get an available port from the configured range"""
        import socket
        
        for port in cls.PORT_RANGE:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        # If no port is available, use a random port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODELS_DIR,
            cls.TRAINED_MODELS_DIR,
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.TEST_RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_summary(cls):
        """Get configuration summary"""
        return {
            'default_port': cls.DEFAULT_PORT,
            'port_range': cls.PORT_RANGE,
            'models_dir': str(cls.MODELS_DIR),
            'data_dir': str(cls.DATA_DIR),
            'debug_mode': cls.DEBUG,
            'api_timeout': cls.API_TIMEOUT,
            'max_batch_size': cls.MAX_BATCH_SIZE
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    ENABLE_CACHING = True


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    API_TIMEOUT = 10
    MAX_BATCH_SIZE = 100


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration class"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])


def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    Config.create_directories()
    
    # Set environment variables if not already set
    if not os.environ.get('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'development'
    
    if not os.environ.get('SECRET_KEY'):
        os.environ['SECRET_KEY'] = Config.SECRET_KEY


if __name__ == "__main__":
    """Test configuration"""
    print("⚙️ Configuration Test")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    # Get configuration
    config_class = get_config()
    print(f"📋 Configuration: {config_class.__name__}")
    print(f"🔧 Debug Mode: {config_class.DEBUG}")
    print(f"🌐 Default Port: {config_class.DEFAULT_PORT}")
    print(f"📁 Models Directory: {config_class.MODELS_DIR}")
    print(f"📊 Data Directory: {config_class.DATA_DIR}")
    
    # Test port availability
    available_port = config_class.get_available_port()
    print(f"🚪 Available Port: {available_port}")
    
    # Configuration summary
    summary = config_class.get_config_summary()
    print(f"\n📊 Configuration Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Configuration test completed!")
