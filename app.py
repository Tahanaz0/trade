from components.web_interface import create_app
from components.config_manager import ConfigManager
from components.strategy_engine import StrategyEngine
from components.performance_tracker import PerformanceTracker

# Initialize components
config_manager = ConfigManager()
performance_tracker = PerformanceTracker()
strategy_engine = StrategyEngine(config_manager, performance_tracker)

# Create Flask app
app = create_app(config_manager, strategy_engine, performance_tracker)

if __name__ == "__main__":
    print("🚀 Starting SMC Strategy Dashboard...")
    print("📡 Access at: http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)



    