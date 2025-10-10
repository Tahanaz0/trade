class ConfigManager:
    def __init__(self):
        self.config = {
            "initial_capital": 50,
            "risk_per_trade": 0.01,
            "coins": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "AVAX/USDT", "XRP/USDT", "LINK/USDT"],
            "timeframe": "1h",
            "daily_trade_limit_per_coin": 5,
            "withdraw_percentage": 0.30,
            "reinvest_percentage": 0.70,
            "volume_multiplier": 1.5,
            "atr_sl_multiplier": 1.0,
            "tp_rr": 3,
            "max_lookforward_bars": 240,
            "testing_months": 1,
            "compounding": True,
            "enable_withdrawal": True,
            "fixed_investment_mode": False,
            "fixed_investment_amount": 10,
            


            "min_bos_strength": 0.5,          # Minimum BOS move %
            "skip_ranging_markets": True,      # Skip ranging conditions
            "time_filter_enabled": True,       # Enable time filter
            "dynamic_tp_sl": True,             # Use adaptive TP/SL
            "min_volume_ratio": 1.2,           # Minimum volume confirmation
        }
    
    def update_config(self, new_settings):
        """Update configuration with new settings"""
        self.config.update(new_settings)
    
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()
    
    def get_setting(self, key):
        """Get specific setting"""
        return self.config.get(key)