"""
Configuration Loader

Loads configuration from YAML files and GCP Secret Manager.
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}

    def load(self, mode: str = 'paper') -> Dict[str, Any]:
        """
        Load configuration for a mode

        Args:
            mode: 'backtest', 'paper', or 'live'

        Returns:
            Configuration dictionary
        """
        # Load base config from YAML
        if self.config_path:
            config_file = os.path.join(self.config_path, f"{mode}.yaml")
        else:
            config_file = f"config/{mode}.yaml"

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            self.config = self._get_default_config(mode)

        # Load secrets from environment or GCP Secret Manager
        self._load_secrets(mode)

        return self.config

    def _load_secrets(self, mode: str):
        """Load secrets from environment variables or GCP Secret Manager"""
        # Try environment variables first
        if 'SUPABASE_URL' in os.environ:
            self.config.setdefault('supabase', {})['url'] = os.environ['SUPABASE_URL']
        if 'SUPABASE_KEY' in os.environ:
            self.config.setdefault('supabase', {})['key'] = os.environ['SUPABASE_KEY']

        # Try GCP Secret Manager if available
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.environ.get('GCP_PROJECT_ID')

            if project_id:
                secrets = [
                    'supabase-url', 'supabase-key',
                    'alpaca-api-key', 'alpaca-api-secret'
                ]

                for secret_name in secrets:
                    try:
                        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                        response = client.access_secret_version(request={"name": name})
                        secret_value = response.payload.data.decode("UTF-8")

                        # Map secret names to config
                        if secret_name == 'supabase-url':
                            self.config.setdefault('supabase', {})['url'] = secret_value
                        elif secret_name == 'supabase-key':
                            self.config.setdefault('supabase', {})['key'] = secret_value
                        elif secret_name == 'alpaca-api-key':
                            self.config.setdefault('broker', {})['api_key'] = secret_value
                        elif secret_name == 'alpaca-api-secret':
                            self.config.setdefault('broker', {})['api_secret'] = secret_value

                    except Exception as e:
                        logger.debug(f"Secret {secret_name} not found in Secret Manager: {e}")

        except ImportError:
            logger.debug("google-cloud-secret-manager not available")
        except Exception as e:
            logger.debug(f"Error loading secrets from GCP: {e}")

    def _get_default_config(self, mode: str) -> Dict[str, Any]:
        """Get default configuration for a mode"""
        defaults = {
            'mode': mode,
            'target_assets': ['SPY', 'QQQ', 'TQQQ'],
            'base_weights': {'SPY': 0.33, 'QQQ': 0.33, 'TQQQ': 0.34},
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'enable_hedging': False,
            'hedge_delta': -0.5,
            'hedge_coverage': 1.0,
            'max_position_size': 0.5,
            'max_portfolio_var': 0.02,
            'max_leverage': 1.0
        }

        if mode == 'backtest':
            defaults.update({
                'data_path': 'data/',
                'output_dir': 'data/results/'
            })
        elif mode in ['paper', 'live']:
            defaults.update({
                'supabase': {
                    'url': os.environ.get('SUPABASE_URL', ''),
                    'key': os.environ.get('SUPABASE_KEY', '')
                },
                'cache_duration': 60
            })

        if mode == 'live':
            defaults.update({
                'broker': {
                    'name': 'alpaca',
                    'api_key': os.environ.get('ALPACA_API_KEY', ''),
                    'api_secret': os.environ.get('ALPACA_API_SECRET', ''),
                    'base_url': os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                }
            })

        return defaults
