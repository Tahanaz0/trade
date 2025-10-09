from flask import Flask, render_template, jsonify, request
import os

def create_app(config_manager, strategy_engine, performance_tracker):
    # Use absolute paths
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    
    template_dir = os.path.join(project_root, 'templates')
    static_dir = os.path.join(project_root, 'static')
    
    print(f"Looking for templates in: {template_dir}")
    print(f"Looking for static files in: {static_dir}")
    
    # Create directories if they don't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)
    
    @app.route('/')
    def index():
        try:
            return render_template('index.html')
        except Exception as e:
            return f"Error loading template: {str(e)}", 500
    
    @app.route('/run', methods=['POST'])
    def run():
        if not strategy_engine.get_running_status():
            settings = request.get_json()
            config_manager.update_config(settings)
            strategy_engine.start_backtest()
        return jsonify({"status": "started"})
    
    @app.route('/data')
    def data():
        return jsonify({
            "trades": strategy_engine.get_trades_data(),
            "summary": performance_tracker.get_summary(),
            "signals": strategy_engine.get_open_signals(),
            "performance": performance_tracker.get_all_performance_data(),
            "is_running": strategy_engine.get_running_status()
        })
    
    return app