from collections import defaultdict

class PerformanceTracker:
    def __init__(self):
        self.summary_data = {}
        self.coin_performance = defaultdict(lambda: {
            "trades": 0, "wins": 0, "losses": 0, "net_profit": 0.0, "total_invested": 0.0
        })
        self.time_performance = {
            "weekly": defaultdict(lambda: {"trades": 0, "net_profit": 0.0}),
            "monthly": defaultdict(lambda: {"trades": 0, "net_profit": 0.0})
        }
    
    def reset(self):
        """Reset all performance data"""
        self.summary_data = {}
        self.coin_performance.clear()
        self.time_performance["weekly"].clear()
        self.time_performance["monthly"].clear()
    
    def update_performance(self, trade):
        """Update performance metrics for a trade"""
        symbol = trade["symbol"]
        
        # Update coin performance
        self.coin_performance[symbol]["trades"] += 1
        self.coin_performance[symbol]["total_invested"] += trade["invested"]
        self.coin_performance[symbol]["net_profit"] += trade["pnl"]
        
        if trade["result"] == "WIN":
            self.coin_performance[symbol]["wins"] += 1
        elif trade["result"] == "LOSS":
            self.coin_performance[symbol]["losses"] += 1
        
        # Update time performance
        trade_date = trade["raw_date"]
        week_key = trade_date.strftime("%Y-W%W")
        month_key = trade_date.strftime("%Y-%m")
        
        self.time_performance["weekly"][week_key]["trades"] += 1
        self.time_performance["weekly"][week_key]["net_profit"] += trade["pnl"]
        self.time_performance["monthly"][month_key]["trades"] += 1
        self.time_performance["monthly"][month_key]["net_profit"] += trade["pnl"]
    
    def set_summary(self, summary):
        """Set summary data"""
        self.summary_data = summary
    
    def get_summary(self):
        """Get summary data"""
        return self.summary_data
    
    def get_coin_performance(self):
        """Get coin performance data"""
        return dict(self.coin_performance)
    
    def get_time_performance(self):
        """Get time performance data"""
        return self.time_performance
    
    def get_all_performance_data(self):
        """Get all performance data"""
        return {
            "coin_performance": self.get_coin_performance(),
            "weekly_performance": dict(self.time_performance["weekly"]),
            "monthly_performance": dict(self.time_performance["monthly"])
        }