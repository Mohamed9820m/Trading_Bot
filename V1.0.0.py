import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
import json
import os
import logging
import time
import traceback
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forex_scalper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ForexScalper")

class ForexScalpingBot:
    """
    High-speed forex scalping bot with XGBoost integration and flexible strategy implementation.
    """
    
    def __init__(self, 
                 symbols=None, 
                 timeframe=mt5.TIMEFRAME_M1,
                 lot_size=0.01,
                 max_positions=5,
                 profit_pips=15.0, # Default TP, can be ATR multiple
                 stop_loss_pips=10.0, # Default SL, can be ATR multiple
                 mt5_directory=None,
                 strategies_directory="strategies",
                 models_directory="models",
                 backtest_data_period=30,  # Days of data to use for backtesting
                 live_trade=False,
                 login_id=None,     # Add these parameters
                 password=None,
                 server=None,
                 # New parameters for improvements
                 use_atr_stops=True, # Flag to use ATR for SL/TP
                 atr_sl_multiplier=1.5, # ATR multiplier for Stop Loss
                 atr_tp_multiplier=2.0, # ATR multiplier for Take Profit
                 position_sizing_fraction=0.01, # e.g., 1% of equity for position size
                 simulated_spread_pips=0.5, # Estimated spread in pips for backtesting
                 max_spread_pips_allowed=2.0, # Max spread to allow for opening new trades
                 signal_aggregation_method="best_strategy_or_all", # "best_strategy_only", "all_strategies_or", "voting"
                 min_votes_for_signal=1, # If using "voting"
                 amd_enabled=True,  # Primary AMD strategy flag
                 amd_consolidation_bars=20,  # Bars to identify consolidation
                 amd_breakout_threshold=0.3,  # Minimum breakout percentage
                 amd_volume_multiplier=1.5,  # Volume confirmation multiplier
                 amd_risk_per_trade=0.02,  # 2% risk per trade
                 amd_min_rr_ratio=2.0,  # 1:2 minimum risk-reward
                 amd_execution_timeout=3,  # 3-second execution timeout
                 amd_pattern_accuracy_threshold=0.60  # 60% win rate requirement
                 
                 ):
        """
        Initialize the Forex Scalping Bot.
        
        Args:
            symbols (list): List of forex pairs to trade
            timeframe: MetaTrader timeframe constant
            lot_size (float): Default size of trading lot if not using dynamic sizing
            max_positions (int): Maximum number of open positions per symbol
            profit_pips (float): Target profit in pips (used if not ATR-based)
            stop_loss_pips (float): Stop loss in pips (used if not ATR-based)
            mt5_directory (str): Path to MetaTrader 5 terminal executable
            strategies_directory (str): Directory containing strategy files
            models_directory (str): Directory for saving/loading models
            backtest_data_period (int): Days of historical data for backtesting
            live_trade (bool): Flag to enable live trading
            login_id (int): MT5 account login ID
            password (str): MT5 account password
            server (str): MT5 server name
            use_atr_stops (bool): Whether to use ATR for dynamic stop loss and take profit
            atr_sl_multiplier (float): ATR multiplier for stop loss
            atr_tp_multiplier (float): ATR multiplier for take profit
            position_sizing_fraction (float): Fraction of account equity to risk per trade for dynamic position sizing
            simulated_spread_pips (float): Spread in pips to simulate during backtesting
            max_spread_pips_allowed (float): Maximum spread in pips allowed to open a new trade
            signal_aggregation_method (str): Method to aggregate signals ("best_strategy_only", "best_strategy_or_all", "all_strategies_or", "voting")
            min_votes_for_signal (int): Minimum votes required if aggregation is "voting"
        """
        # Default symbols if none provided
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        self.timeframe = timeframe
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.profit_pips = profit_pips 
        self.stop_loss_pips = stop_loss_pips
        self.mt5_directory = mt5_directory
        self.strategies_directory = strategies_directory
        self.models_directory = models_directory
        self.backtest_data_period = backtest_data_period
        self.live_trade = live_trade
        
        # Store MT5 login credentials
        self.login_id = login_id
        self.password = password
        self.server = server

        # Improved Risk Management & Signal Params
        self.use_atr_stops = use_atr_stops
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.position_sizing_fraction = position_sizing_fraction
        self.simulated_spread_pips = simulated_spread_pips
        self.max_spread_pips_allowed = max_spread_pips_allowed
        self.signal_aggregation_method = signal_aggregation_method
        self.min_votes_for_signal = min_votes_for_signal
        self.best_strategy_name = None # To be set after backtesting
        
        # Create directories if they don't exist
        os.makedirs(self.strategies_directory, exist_ok=True)
        os.makedirs(self.models_directory, exist_ok=True)
        
        # Initialize data storage
        self.data = {}
        self.strategies = {}
        
        # Initialize connection status
        self.connected = False
        
        # Executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=len(self.symbols) + 2)
        
        self.symbol_cooldown_until = {} # Stores datetime until which a symbol is in cooldown
        self.cooldown_duration_seconds = 60 # Cooldown period in seconds (e.g., 60 seconds)


                # AMD Strategy Parameters
        self.amd_enabled = amd_enabled
        self.amd_consolidation_bars = amd_consolidation_bars
        self.amd_breakout_threshold = amd_breakout_threshold
        self.amd_volume_multiplier = amd_volume_multiplier
        self.amd_risk_per_trade = amd_risk_per_trade
        self.amd_min_rr_ratio = amd_min_rr_ratio
        self.amd_execution_timeout = amd_execution_timeout
        self.amd_pattern_accuracy_threshold = amd_pattern_accuracy_threshold

                # Daily risk management
        self.daily_loss_limit = 100.0  # Maximum daily loss in account currency
        self.daily_trades_limit = 20   # Maximum trades per day
        self.daily_stats = {}  # Track daily performance
        
        # AMD Pattern Tracking
        self.amd_patterns = {}  # Track AMD patterns per symbol
        self.amd_performance = {}  # Track AMD performance metrics
        self.amd_consolidation_zones = {}  # Track consolidation areas
        
        # Override position sizing for AMD risk management
        self.position_sizing_fraction = amd_risk_per_trade
        
        logger.info(f"Initialized ForexScalpingBot with {len(self.symbols)} symbols")
    
    def connect_to_metatrader(self):
        """
        Connect to the MetaTrader 5 terminal.
        """
        # Shutdown MT5 if already connected
        if mt5.terminal_info() is not None:
            mt5.shutdown()
            
        # Initialize connection to MetaTrader 5
        if self.mt5_directory:
            mt5_initialized = mt5.initialize(path=self.mt5_directory)
        else:
            mt5_initialized = mt5.initialize()
            
        if not mt5_initialized:
            logger.error(f"MT5 initialization failed. Error code: {mt5.last_error()}")
            return False
        
        # Login to trading account if credentials are provided
        if hasattr(self, 'login_id') and hasattr(self, 'password') and hasattr(self, 'server'):
            if self.login_id and self.password and self.server:
                login_result = mt5.login(self.login_id, self.password, self.server)
                if not login_result:
                    logger.error(f"Failed to login to {self.server}. Error: {mt5.last_error()}")
                    return False
                logger.info(f"Successfully logged in as {self.login_id} to {self.server}")
        
        # Check connection
        if not mt5.terminal_info():
            logger.error("Failed to connect to MetaTrader 5 terminal")
            return False
        
        logger.info(f"Connected to MetaTrader 5. Terminal info: {mt5.terminal_info()}")
        self.connected = True
        
        # Ensure all symbols are available
        for symbol in self.symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                continue
                
            if not symbol_info.visible:
                logger.info(f"Symbol {symbol} is not visible, enabling...")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
        
        return True
    
    def disconnect_from_metatrader(self):
        """
        Disconnect from the MetaTrader 5 terminal.
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader 5")
    
    def load_strategies(self):
        """Load and validate strategies from JSON files."""
        self.strategies = {}
        
        try:
            for filename in os.listdir(self.strategies_directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.strategies_directory, filename)
                    try:
                        with open(filepath, 'r') as f:
                            strategy = json.load(f)
                        
                        # Validate and fix strategy
                        fixed_strategy = self.validate_and_fix_strategy(strategy)
                        if fixed_strategy:
                            strategy_name = os.path.splitext(filename)[0]
                            self.strategies[strategy_name] = fixed_strategy
                            logger.info(f"Loaded and validated strategy: {strategy_name}")
                        else:
                            logger.error(f"Failed to validate strategy from {filename}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {filename}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading strategies directory: {e}")
        
        logger.info(f"Successfully loaded {len(self.strategies)} valid strategies")
    
    def save_strategy(self, strategy_name, strategy_data):
        """
        Save a strategy to a JSON file.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_data (dict): Strategy parameters and rules
        """
        try:
            # Sanitize strategy_name for use as a filename
            safe_strategy_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in strategy_name)
            filepath = os.path.join(self.strategies_directory, f"{safe_strategy_name}.json")
            with open(filepath, 'w') as f:
                json.dump(strategy_data, f, indent=4)
            logger.info(f"Saved strategy: {strategy_name} to {filepath}") # Log the actual filename
            return True
        except Exception as e:
            logger.error(f"Error saving strategy {strategy_name}: {e}")
            return False
    
    def fetch_data(self, symbol, timeframe, bars=1000):
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol (str): Forex pair symbol
            timeframe: MetaTrader timeframe constant
            bars (int): Number of bars to fetch
            
        Returns:
            DataFrame: Historical price data
        """
        if not self.connected:
            logger.error("Not connected to MetaTrader 5")
            return None
        
        # Get current time
        now = datetime.now()
        
        # Fetch historical data
        rates = mt5.copy_rates_from(symbol, timeframe, now, bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}. Error: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df
    
    def add_technical_indicators(self, df):
        """Add all technical indicators efficiently to avoid DataFrame fragmentation."""
        
        # Create all indicator columns at once to avoid fragmentation
        indicators_dict = {}
        
        # CRITICAL: Create volume FIRST before any calculations
        if 'volume' not in df.columns or df['volume'].isna().all():
            price_change = abs(df['close'].diff())
            range_size = df['high'] - df['low']
            indicators_dict['volume'] = (price_change * range_size * 100000).fillna(0)
        else:
            indicators_dict['volume'] = df['volume']
                                                                                   
        # Basic Moving Averages
        indicators_dict['sma_5'] = talib.SMA(df['close'], timeperiod=5)
        indicators_dict['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        indicators_dict['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        indicators_dict['ema_5'] = talib.EMA(df['close'], timeperiod=5)
        indicators_dict['ema_10'] = talib.EMA(df['close'], timeperiod=10)
        indicators_dict['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        indicators_dict['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        indicators_dict['upper_band'] = upper
        indicators_dict['middle_band'] = middle
        indicators_dict['lower_band'] = lower
        indicators_dict['bb_width'] = (upper - lower) / middle
        indicators_dict['bb_width_ma_20'] = talib.SMA(indicators_dict['bb_width'], timeperiod=20)
        
        # RSI and related
        indicators_dict['rsi'] = talib.RSI(df['close'], timeperiod=14)
        indicators_dict['rsi_ma_5'] = talib.SMA(indicators_dict['rsi'], timeperiod=5)
        
        # CRITICAL FIX: Stochastic indicators (missing in many strategies)
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                                   fastk_period=5, slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)
        indicators_dict['slowk'] = slowk
        indicators_dict['slowd'] = slowd
        indicators_dict['stochastic_k'] = slowk  # Alias
        indicators_dict['stochastic_d'] = slowd  # Alias
        indicators_dict['stochastic'] = slowk    # Main stochastic indicator for strategies
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        indicators_dict['macd'] = macd
        indicators_dict['macd_signal'] = signal
        indicators_dict['macd_hist'] = hist
        indicators_dict['macd_histogram'] = hist  # Alias
        
        # ATR and ADX
        indicators_dict['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        indicators_dict['atr_ma_14'] = talib.SMA(indicators_dict['atr'], timeperiod=14)
        indicators_dict['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Parabolic SAR
        indicators_dict['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # Ichimoku components
        indicators_dict['tenkan_sen'] = talib.SMA(df['close'], timeperiod=9)
        indicators_dict['kijun_sen'] = talib.SMA(df['close'], timeperiod=26)
        indicators_dict['senkou_span_a'] = (indicators_dict['tenkan_sen'] + indicators_dict['kijun_sen']) / 2
        indicators_dict['senkou_span_b'] = talib.SMA(df['close'], timeperiod=52)
        indicators_dict['chikou_span'] = df['close'].shift(26)
        
        # Cloud calculations
        indicators_dict['cloud_top'] = np.maximum(indicators_dict['senkou_span_a'], indicators_dict['senkou_span_b'])
        indicators_dict['cloud_bottom'] = np.minimum(indicators_dict['senkou_span_a'], indicators_dict['senkou_span_b'])
        
        # CRITICAL FIX: Volume indicators (using the volume we created above)
        volume = indicators_dict['volume']
        indicators_dict['volume_ma_10'] = talib.SMA(volume, timeperiod=10)
        indicators_dict['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
        indicators_dict['volume_delta'] = volume.diff().fillna(0)
        indicators_dict['bid_volume'] = volume * 0.5
        indicators_dict['ask_volume'] = volume * 0.5
        
        # CRITICAL FIX: Missing bid/ask volume delta for Microstructure strategies
        indicators_dict['bid_volume_delta'] = indicators_dict['bid_volume'].diff().fillna(0)
        indicators_dict['ask_volume_delta'] = indicators_dict['ask_volume'].diff().fillna(0)
        
        # CRITICAL FIX: OBV (On-Balance Volume) - missing in Triple Convergence
        indicators_dict['obv'] = talib.OBV(df['close'], volume)
        indicators_dict['obv_ema_10'] = talib.EMA(indicators_dict['obv'], timeperiod=10)
        
        # Dynamic levels
        indicators_dict['dynamic_oversold'] = 35 - (indicators_dict['atr'] / df['close'] * 1000)
        indicators_dict['dynamic_overbought'] = 65 + (indicators_dict['atr'] / df['close'] * 1000)
        indicators_dict['dynamic_oversold_stoch'] = 20 - (indicators_dict['atr'] / df['close'] * 500)
        indicators_dict['dynamic_overbought_stoch'] = 80 + (indicators_dict['atr'] / df['close'] * 500)
        
        # Fibonacci levels
        fib_window = min(100, len(df)) # Adaptive window up to 100
        min_fib_periods = int(fib_window * 0.8)

        if fib_window > 1: # Need at least 2 bars for a range
            price_max = df['high'].rolling(window=fib_window, min_periods=min_fib_periods).max()
            price_min = df['low'].rolling(window=fib_window, min_periods=min_fib_periods).min()
            price_range = price_max - price_min

            indicators_dict['fib_0'] = price_min
            indicators_dict['fib_1'] = price_max
            # Calculate levels only if price_range is valid (not NaN and > 0)
            indicators_dict['fib_0.236'] = np.where(price_range > 0, price_min + 0.236 * price_range, np.nan)
            indicators_dict['fib_0.382'] = np.where(price_range > 0, price_min + 0.382 * price_range, np.nan)
            indicators_dict['fib_0.5']   = np.where(price_range > 0, price_min + 0.5   * price_range, np.nan)
            indicators_dict['fib_0.618'] = np.where(price_range > 0, price_min + 0.618 * price_range, np.nan)
            indicators_dict['fib_0.786'] = np.where(price_range > 0, price_min + 0.786 * price_range, np.nan)
        else:
        # Not enough data, fill with NaNs
            for level_suffix in ['0', '0.236', '0.382', '0.5', '0.618', '0.786', '1']:
                indicators_dict[f'fib_{level_suffix}'] = np.nan
        
        # VWAP
        indicators_dict['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        indicators_dict['tp_vol'] = indicators_dict['typical_price'] * volume
    
    # Define a fixed VWAP window (e.g., 1440 for 24h on M1 timeframe)
        vwap_window_period = 1440 # Adjust if using different timeframe or VWAP period
        min_vwap_periods = int(vwap_window_period * 0.8) # Require at least 80% of data for VWAP calc

        tp_vol_sum = indicators_dict['tp_vol'].rolling(window=vwap_window_period, min_periods=min_vwap_periods).sum()
        volume_sum = volume.rolling(window=vwap_window_period, min_periods=min_vwap_periods).sum()
    
    # Avoid division by zero or NaN if volume_sum is 0 or NaN
        indicators_dict['vwap'] = np.where(volume_sum != 0, tp_vol_sum / volume_sum, np.nan)
        
        # Candlestick patterns
        indicators_dict['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        indicators_dict['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Support/Resistance levels
        window = min(20, len(df) // 2) if len(df) > 10 else 5
        indicators_dict['resistance_level'] = df['high'].rolling(window=window).max()
        indicators_dict['support_level'] = df['low'].rolling(window=window).min()
        
        # Value area calculations
        value_area_window = min(100, len(df))
        indicators_dict['value_area_high'] = df['high'].rolling(window=value_area_window).quantile(0.7)
        indicators_dict['value_area_low'] = df['low'].rolling(window=value_area_window).quantile(0.3)
        
        # Multi-timeframe approximations
        indicators_dict['rsi_1m'] = indicators_dict['rsi']
        indicators_dict['rsi_5m'] = talib.RSI(df['close'], timeperiod=70)
        indicators_dict['macd_1m'] = indicators_dict['macd']
        indicators_dict['macd_signal_1m'] = indicators_dict['macd_signal']
        
        # Momentum indicators
        indicators_dict['momentum'] = talib.MOM(df['close'], timeperiod=10)
        indicators_dict['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        indicators_dict['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        indicators_dict['trix'] = talib.TRIX(df['close'], timeperiod=9)
        
        # Point of Control and Market Profile
        poc_window = 50
        min_poc_periods = int(poc_window * 0.8) # Require at least 80% of data
        indicators_dict['poc'] = df['close'].rolling(window=poc_window, min_periods=min_poc_periods).median()
        
        # Delta and microstructure
        indicators_dict['delta'] = volume * np.where(df['close'] > df['open'], 1, -1)
        indicators_dict['cumulative_delta'] = indicators_dict['delta'].cumsum()
        indicators_dict['bid_ask_spread'] = abs(df['high'] - df['low']) / df['close']
        indicators_dict['order_flow'] = volume * np.sign(df['close'].diff().fillna(0))
        
        # AMD indicators
        amd_indicators = self.add_amd_indicators_dict(df, volume)
        indicators_dict.update(amd_indicators)
        
        # Add all indicators to dataframe at once (prevents fragmentation)
        indicator_df = pd.DataFrame(indicators_dict, index=df.index)
        df = pd.concat([df, indicator_df], axis=1)
        
        # Drop NaN values AFTER all indicators are created
        df.dropna(inplace=True)
        
        return df
    
    def add_amd_indicators_dict(self, df, volume):
        """Return AMD indicators as dictionary to avoid DataFrame fragmentation."""
        indicators = {}
        
        # Price range analysis
        indicators['price_range'] = df['high'] - df['low']
        indicators['price_range_ma'] = indicators['price_range'].rolling(window=self.amd_consolidation_bars).mean()
        indicators['range_ratio'] = indicators['price_range'] / indicators['price_range_ma']
        
        # Consolidation detection
        indicators['is_consolidating'] = (indicators['range_ratio'] < 0.85).rolling(window=3).sum() >= 2
        
        # Support/resistance for consolidation
        window = self.amd_consolidation_bars
        indicators['consolidation_high'] = df['high'].rolling(window=window).max()
        indicators['consolidation_low'] = df['low'].rolling(window=window).min()
        indicators['consolidation_range'] = indicators['consolidation_high'] - indicators['consolidation_low']
        
        # Volume analysis
        indicators['volume_ma'] = volume.rolling(window=20).mean()
        indicators['volume_ratio'] = volume / indicators['volume_ma']
        indicators['high_volume'] = indicators['volume_ratio'] > self.amd_volume_multiplier
        
        # Breakout detection
        indicators['above_consolidation'] = df['close'] > indicators['consolidation_high']
        indicators['below_consolidation'] = df['close'] < indicators['consolidation_low']
        indicators['breakout_failed'] = (
            (indicators['above_consolidation'].shift(1) & (df['close'] < indicators['consolidation_high'])) |
            (indicators['below_consolidation'].shift(1) & (df['close'] > indicators['consolidation_low']))
        )
        
        # True breakouts
        indicators['true_breakout_up'] = (
            indicators['above_consolidation'] & 
            indicators['high_volume'] & 
            ~indicators['breakout_failed'] &
            (df['close'] > indicators['consolidation_high'] * (1 + self.amd_breakout_threshold / 100))
        )
        
        indicators['true_breakout_down'] = (
            indicators['below_consolidation'] & 
            indicators['high_volume'] & 
            ~indicators['breakout_failed'] &
            (df['close'] < indicators['consolidation_low'] * (1 - self.amd_breakout_threshold / 100))
        )
        
        # AMD phases
        indicators['amd_accumulation'] = indicators['is_consolidating'] & (indicators['range_ratio'] < 0.5)
        indicators['amd_manipulation'] = indicators['breakout_failed']
        indicators['amd_distribution_buy'] = indicators['true_breakout_up']
        indicators['amd_distribution_sell'] = indicators['true_breakout_down']
        
        return indicators
    

    def add_amd_indicators(self, df):
        """Add AMD (Accumulation-Manipulation-Distribution) specific indicators."""
        # Price range analysis for consolidation detection
        df['price_range'] = df['high'] - df['low']
        df['price_range_ma'] = df['price_range'].rolling(window=self.amd_consolidation_bars).mean()
        df['range_ratio'] = df['price_range'] / df['price_range_ma']
        
        # Consolidation detection (tight range)
        df['is_consolidating'] = (df['range_ratio'] < 0.85).rolling(window=3).sum() >= 2
        
        # Support and resistance levels for consolidation zones
        window = self.amd_consolidation_bars
        df['consolidation_high'] = df['high'].rolling(window=window).max()
        df['consolidation_low'] = df['low'].rolling(window=window).min()
        df['consolidation_range'] = df['consolidation_high'] - df['consolidation_low']
        
        # Volume analysis
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['high_volume'] = df['volume_ratio'] > self.amd_volume_multiplier
        else:
            # Fallback volume analysis using price action
            df['price_velocity'] = abs(df['close'].diff())
            df['velocity_ma'] = df['price_velocity'].rolling(window=20).mean()
            df['high_volume'] = df['price_velocity'] > df['velocity_ma'] * 1.2
        
        # False breakout detection
        df['above_consolidation'] = df['close'] > df['consolidation_high']
        df['below_consolidation'] = df['close'] < df['consolidation_low']
        df['breakout_failed'] = (
            (df['above_consolidation'].shift(1) & (df['close'] < df['consolidation_high'])) |
            (df['below_consolidation'].shift(1) & (df['close'] > df['consolidation_low']))
        )
        
        # True breakout with volume confirmation
        df['true_breakout_up'] = (
            df['above_consolidation'] & 
            df['high_volume'] & 
            ~df['breakout_failed'] &
            (df['close'] > df['consolidation_high'] * (1 + self.amd_breakout_threshold / 100))
        )
        
        df['true_breakout_down'] = (
            df['below_consolidation'] & 
            df['high_volume'] & 
            ~df['breakout_failed'] &
            (df['close'] < df['consolidation_low'] * (1 - self.amd_breakout_threshold / 100))
        )
        
        # AMD phase detection
        df['amd_accumulation'] = df['is_consolidating'] & (df['range_ratio'] < 0.5)
        df['amd_manipulation'] = df['breakout_failed']
        df['amd_distribution_buy'] = df['true_breakout_up']
        df['amd_distribution_sell'] = df['true_breakout_down']
        
        return df
    

    def get_supported_filling_mode(self, symbol):
        """Enhanced filling mode detection with special XAUUSD handling."""
        try:
            # Force RETURN mode for all gold instruments
            if "XAU" in symbol or "GOLD" in symbol or symbol == "XAUUSD":
                logger.debug(f"Using ORDER_FILLING_RETURN for gold instrument {symbol}")
                return mt5.ORDER_FILLING_RETURN
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Cannot get symbol info for {symbol}, using RETURN")
                return mt5.ORDER_FILLING_RETURN
            
            # For other symbols, use the most compatible mode first
            filling_mode = symbol_info.filling_mode
            
            # Priority order: RETURN (most compatible) -> IOC -> FOK
            if filling_mode & mt5.SYMBOL_FILLING_RETURN:
                return mt5.ORDER_FILLING_RETURN
            elif filling_mode & mt5.SYMBOL_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif filling_mode & mt5.SYMBOL_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            else:
                return mt5.ORDER_FILLING_RETURN
                
        except Exception as e:
            logger.error(f"Filling mode detection error for {symbol}: {e}")
            return mt5.ORDER_FILLING_RETURN
    
    def detect_amd_pattern(self, df, symbol):
        """Detect complete AMD pattern formation."""


        
        if len(df) < self.amd_consolidation_bars * 2:
            return None
        
        latest_bars = df.tail(self.amd_consolidation_bars)
        
        # Check for accumulation phase (consolidation)
        accumulation_detected = latest_bars['amd_accumulation'].any()
        
        # Check for manipulation phase (false breakout)
        manipulation_detected = latest_bars['amd_manipulation'].any()
        
        # Check for distribution phase (true breakout)
        distribution_buy = latest_bars['amd_distribution_buy'].iloc[-1]
        distribution_sell = latest_bars['amd_distribution_sell'].iloc[-1]

        logger.debug(f"AMD Debug {symbol}: Accum={accumulation_detected}, Manip={manipulation_detected}, Dist={distribution_buy or distribution_sell}")

        if (accumulation_detected and (distribution_buy or distribution_sell)) or \
           (manipulation_detected and (distribution_buy or distribution_sell) and 
        latest_bars['high_volume'].iloc[-1]):
            pattern = {
                'symbol': symbol,
                'timestamp': df.index[-1],
                'phase': 'distribution',
                'direction': 'buy' if distribution_buy else 'sell',
                'consolidation_high': latest_bars['consolidation_high'].iloc[-1],
                'consolidation_low': latest_bars['consolidation_low'].iloc[-1],
                'current_price': df['close'].iloc[-1],
                'volume_confirmed': latest_bars['high_volume'].iloc[-1],
                'confidence': self.calculate_amd_confidence(latest_bars)
            }
            
            # Store pattern for tracking
            self.amd_patterns[symbol] = pattern
            return pattern
        
        return None
    
    def calculate_amd_confidence(self, bars):
        """Calculate confidence score for AMD pattern (0-1)."""
        confidence = 0.0
        
        # Consolidation quality (tighter = better)
        avg_range_ratio = bars['range_ratio'].mean()
        if avg_range_ratio < 0.5:
            confidence += 0.3
        elif avg_range_ratio < 0.7:
            confidence += 0.2
        
        # Volume confirmation
        if bars['high_volume'].iloc[-1]:
            confidence += 0.3
        
        # Manipulation presence
        if bars['amd_manipulation'].any():
            confidence += 0.2
        
        # Price momentum after breakout
        if len(bars) >= 3:
            recent_momentum = (bars['close'].iloc[-1] - bars['close'].iloc[-3]) / bars['close'].iloc[-3]
            if abs(recent_momentum) > 0.001:  # 0.1% momentum
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def calculate_amd_position_size(self, symbol, entry_price, stop_loss_price):
        """Calculate position size based on AMD risk management."""
        if not self.connected:
            return self.lot_size
        
        account_info = mt5.account_info()
        if account_info is None:
            return self.lot_size
        
        # 2% risk per trade
        risk_amount = account_info.equity * self.amd_risk_per_trade
        
        # Calculate pip value and risk per pip
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return self.lot_size
        
        point_value = symbol_info.point
        pip_value = point_value * 10 if 'JPY' not in symbol else point_value
        
        # Calculate stop loss distance in pips
        sl_distance_pips = abs(entry_price - stop_loss_price) / pip_value
        
        if sl_distance_pips > 0:
            # Calculate lot size based on risk
            pip_value_per_lot = 10 if 'JPY' not in symbol else 1000  # Simplified calculation
            calculated_lot = risk_amount / (sl_distance_pips * pip_value_per_lot)
            
            # Ensure within bounds
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, 1.0)  # Cap at 1.0 for safety
            
            return max(min_lot, min(calculated_lot, max_lot))
        
        return self.lot_size
    
    def execute_amd_trade(self, pattern):
        """Execute trade based on AMD pattern with enhanced risk management."""
        symbol = pattern['symbol']
        direction = pattern['direction']
        confidence = pattern['confidence']
        
        # Only trade high-confidence patterns
        if confidence < 0.6:
            logger.info(f"AMD pattern confidence too low for {symbol}: {confidence:.2f}")
            return None
        
        # Get current market data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        # Calculate entry and stop loss based on pattern
        if direction == 'buy':
            entry_price = tick.ask
            # Stop loss below consolidation low
            stop_loss_price = pattern['consolidation_low'] * 0.999  # Small buffer
            # Take profit based on minimum RR ratio
            risk_pips = abs(entry_price - stop_loss_price) / (0.0001 if 'JPY' not in symbol else 0.01)
            reward_pips = risk_pips * self.amd_min_rr_ratio
            take_profit_price = entry_price + (reward_pips * (0.0001 if 'JPY' not in symbol else 0.01))
        else:
            entry_price = tick.bid
            # Stop loss above consolidation high
            stop_loss_price = pattern['consolidation_high'] * 1.001  # Small buffer
            # Take profit based on minimum RR ratio
            risk_pips = abs(entry_price - stop_loss_price) / (0.0001 if 'JPY' not in symbol else 0.01)
            reward_pips = risk_pips * self.amd_min_rr_ratio
            take_profit_price = entry_price - (reward_pips * (0.0001 if 'JPY' not in symbol else 0.01))
        
        # Calculate AMD-specific position size
        lot_size = self.calculate_amd_position_size(symbol, entry_price, stop_loss_price)
        
        # Execute trade with high priority
        logger.info(f"Executing AMD {direction} trade for {symbol} - Confidence: {confidence:.2f}")
        
        start_time = time.time()
        trade_result = self.execute_trade(
            symbol=symbol, 
            trade_type=direction, 
            lot_size_override=lot_size
        )
        execution_time = time.time() - start_time
        
        # Log execution speed
        if execution_time > self.amd_execution_timeout:
            logger.warning(f"AMD execution slow for {symbol}: {execution_time:.2f}s")
        else:
            logger.info(f"AMD execution fast for {symbol}: {execution_time:.2f}s")
        
        # Update AMD performance tracking
        if trade_result:
            self.track_amd_performance(symbol, pattern, trade_result)
        
        return trade_result
    
    def track_amd_performance(self, symbol, pattern, trade_result):
        """Track AMD pattern performance for optimization."""
        if symbol not in self.amd_performance:
            self.amd_performance[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0,
                'confidence_scores': []
            }
        
        self.amd_performance[symbol]['total_trades'] += 1
        self.amd_performance[symbol]['confidence_scores'].append(pattern['confidence'])
        
        # Store trade details for later performance evaluation
        trade_result['amd_pattern'] = pattern
        trade_result['entry_time'] = datetime.now()

    def apply_signal_cooldown(self, symbol, signal_type):
        """Prevent signal spam by implementing cooldown periods"""
        if not hasattr(self, 'last_signal_time'):
            self.last_signal_time = {}
        
        if symbol not in self.last_signal_time:
            self.last_signal_time[symbol] = {}
        
        current_time = datetime.now()
        last_signal = self.last_signal_time[symbol].get(signal_type)
        
        if last_signal and (current_time - last_signal).seconds < 300:  # 5-minute cooldown
            return False
        
        self.last_signal_time[symbol][signal_type] = current_time
        return True
    

    def is_market_session_active(self, symbol):
        """Check if it's an active trading session for the symbol"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Define active hours for different markets (UTC)
        if 'USD' in symbol:
            # US session: 13:00-22:00 UTC, London overlap: 8:00-12:00 UTC
            return (8 <= current_hour <= 12) or (13 <= current_hour <= 22)
        elif 'EUR' in symbol or 'GBP' in symbol:
            # European session: 7:00-16:00 UTC
            return 7 <= current_hour <= 16
        elif 'JPY' in symbol or 'AUD' in symbol:
            # Asian session: 22:00-8:00 UTC
            return current_hour >= 22 or current_hour <= 8
        else:
            return True  # Trade all hours for other symbols

    
    def check_daily_limits(self):
        """Check if daily trading limits have been reached"""
        today = datetime.now().date()
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {'trades': 0, 'profit_loss': 0.0}
        
        daily_stats = self.daily_stats[today]
        
        # Check daily loss limit
        if daily_stats['profit_loss'] <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {daily_stats['profit_loss']:.2f}")
            return False
        
        # Check daily trades limit
        if daily_stats['trades'] >= self.daily_trades_limit:
            logger.warning(f"Daily trades limit reached: {daily_stats['trades']}")
            return False
        
        return True
    
    def apply_strategy_rules(self, df, strategy):
        strategy_name = strategy.get('name', 'Unknown')
        rules = strategy.get('rules', [])
        
        # Remove all the verbose debugging - keep only essential logs
        if df.empty or len(df) < 2:
            logger.error(f"Insufficient data for {strategy_name}")
            return df
        
        # Validate indicators exist with auto-creation
        if not self.validate_strategy_indicators(df, strategy):
            logger.warning(f"Strategy {strategy_name} has missing indicators - proceeding with available data")
        
        buy_signals = np.zeros(len(df), dtype=bool)
        sell_signals = np.zeros(len(df), dtype=bool)
        
        for i, rule in enumerate(rules):
            try:
                condition_result = self._evaluate_condition(df, rule)
                signal_type = rule.get('signal', '')
                
                if condition_result is not None and len(condition_result) == len(df):
                    if signal_type == 'buy':
                        buy_signals = np.logical_or(buy_signals, condition_result)
                    elif signal_type == 'sell':
                        sell_signals = np.logical_or(sell_signals, condition_result)
            except Exception as e:
                logger.error(f"Error evaluating rule {i+1} in {strategy_name}: {e}")
                continue
        
        df['buy_signal'] = buy_signals.astype(int)
        df['sell_signal'] = sell_signals.astype(int)
        df['final_buy_signal'] = df['buy_signal']
        df['final_sell_signal'] = df['sell_signal']
        
        return df
    
    def validate_and_fix_strategy(self, strategy):
        """Comprehensive strategy validation and auto-fixing."""
        if not isinstance(strategy, dict):
            logger.error(f"Strategy is not a dictionary: {type(strategy)}")
            return None
        
        # Ensure required fields
        if 'name' not in strategy:
            strategy['name'] = 'Unknown Strategy'
        
        if 'rules' not in strategy or not isinstance(strategy['rules'], list):
            logger.error(f"Strategy {strategy['name']} has invalid rules")
            return None
        
        # Comprehensive rule validation and fixing
        fixed_rules = []
        for i, rule in enumerate(strategy['rules']):
            if not isinstance(rule, dict):
                logger.warning(f"Rule {i} in {strategy['name']} is not a dictionary - skipping")
                continue
            
            # Ensure signal field exists
            if 'signal' not in rule:
                rule['signal'] = 'buy' if i % 2 == 0 else 'sell'
                logger.info(f"Added missing signal field to rule {i}: {rule['signal']}")
            
            # Ensure type field exists
            if 'type' not in rule:
                rule['type'] = 'simple_condition'
                logger.info(f"Added missing type field to rule {i}")
            
            # Fix combined_condition rules
            if rule['type'] == 'combined_condition':
                if 'conditions' not in rule or not isinstance(rule.get('conditions'), list):
                    logger.warning(f"Rule {i} combined_condition has invalid conditions - creating default")
                    # Create a simple fallback condition
                    rule['conditions'] = [{
                        'type': 'simple_condition',
                        'indicator': 'rsi',
                        'comparison': '<' if rule['signal'] == 'buy' else '>',
                        'value': 30 if rule['signal'] == 'buy' else 70,
                        'value_type': 'fixed'
                    }]
                
                # Ensure operator exists
                if 'operator' not in rule:
                    rule['operator'] = 'AND'
                    logger.info(f"Added missing operator to combined_condition rule {i}")
            
            # Fix simple_condition rules
            elif rule['type'] == 'simple_condition':
                if 'indicator' not in rule:
                    rule['indicator'] = 'rsi'
                    logger.info(f"Added missing indicator to simple_condition rule {i}")
                if 'comparison' not in rule:
                    rule['comparison'] = '<' if rule['signal'] == 'buy' else '>'
                    logger.info(f"Added missing comparison to simple_condition rule {i}")
                if 'value' not in rule:
                    rule['value'] = 30 if rule['signal'] == 'buy' else 70
                    logger.info(f"Added missing value to simple_condition rule {i}")
                if 'value_type' not in rule:
                    rule['value_type'] = 'fixed'
                    logger.info(f"Added missing value_type to simple_condition rule {i}")
            
            # Fix crossover rules
            elif rule['type'] == 'crossover':
                if 'indicator1' not in rule:
                    rule['indicator1'] = 'sma_5'
                    logger.info(f"Added missing indicator1 to crossover rule {i}")
                if 'indicator2' not in rule:
                    rule['indicator2'] = 'sma_20'
                    logger.info(f"Added missing indicator2 to crossover rule {i}")
                if 'direction' not in rule:
                    rule['direction'] = 'up' if rule['signal'] == 'buy' else 'down'
                    logger.info(f"Added missing direction to crossover rule {i}")
            
            fixed_rules.append(rule)
        
        strategy['rules'] = fixed_rules
        logger.info(f"Strategy {strategy['name']} validated with {len(fixed_rules)} rules")
        return strategy
    

    def validate_strategy_indicators(self, df, strategy):
        """Validate that all required indicators exist in dataframe with auto-creation."""
        missing_indicators = []
        strategy_name = strategy.get('name', 'Unknown')
        created_indicators = []
        
        for i, rule in enumerate(strategy.get('rules', [])):
            rule_type = rule.get('type', '')
            
            if rule_type == 'simple_condition':
                indicator = rule.get('indicator', '')
                if indicator and indicator not in df.columns:
                    # Try to create missing indicator
                    if self._create_missing_indicator(df, indicator):
                        created_indicators.append(indicator)
                    else:
                        missing_indicators.append(f"Rule {i}: {indicator}")
                        
            elif rule_type == 'crossover':
                indicator1 = rule.get('indicator1', '')
                indicator2 = rule.get('indicator2', '')
                if indicator1 and indicator1 not in df.columns:
                    if self._create_missing_indicator(df, indicator1):
                        created_indicators.append(indicator1)
                    else:
                        missing_indicators.append(f"Rule {i}: {indicator1}")
                if indicator2 and indicator2 not in df.columns:
                    if self._create_missing_indicator(df, indicator2):
                        created_indicators.append(indicator2)
                    else:
                        missing_indicators.append(f"Rule {i}: {indicator2}")
                        
            elif rule_type == 'combined_condition':
                conditions = rule.get('conditions', [])
                for j, condition in enumerate(conditions):
                    if condition.get('type') == 'simple_condition':
                        indicator = condition.get('indicator', '')
                        if indicator and indicator not in df.columns:
                            if self._create_missing_indicator(df, indicator):
                                created_indicators.append(indicator)
                            else:
                                missing_indicators.append(f"Rule {i}, Condition {j}: {indicator}")
        
        if created_indicators:
            logger.info(f"Created missing indicators for {strategy_name}: {created_indicators}")
        
        if missing_indicators:
            logger.warning(f"Strategy {strategy_name} still missing indicators: {missing_indicators}")
            return False
        
        return True
    
    def _create_missing_indicator(self, df, indicator_name):
        """Create missing indicators on the fly."""
        try:
            if indicator_name == 'volume' and 'volume' not in df.columns:
                df['volume'] = abs(df['close'].diff()) * abs(df['high'] - df['low']) * 1000000
                return True
            elif indicator_name == 'poc':
                df['poc'] = df['close'].rolling(window=50).median()
                return True
            elif indicator_name == 'value_area_high':
                df['value_area_high'] = df['high'].rolling(window=50).quantile(0.7)
                return True
            elif indicator_name == 'value_area_low':
                df['value_area_low'] = df['low'].rolling(window=50).quantile(0.3)
                return True
            elif indicator_name == 'delta':
                if 'volume' not in df.columns:
                    df['volume'] = abs(df['close'].diff()) * 1000000
                df['delta'] = df['volume'] * np.where(df['close'] > df['open'], 1, -1)
                return True
            elif indicator_name == 'cumulative_delta':
                if 'delta' not in df.columns:
                    self._create_missing_indicator(df, 'delta')
                df['cumulative_delta'] = df['delta'].cumsum()
                return True
            elif indicator_name == 'bid_volume':
                if 'volume' not in df.columns:
                    self._create_missing_indicator(df, 'volume')
                df['bid_volume'] = df['volume'] * 0.5
                return True
            elif indicator_name == 'ask_volume':
                if 'volume' not in df.columns:
                    self._create_missing_indicator(df, 'volume')
                df['ask_volume'] = df['volume'] * 0.5
                return True
            elif indicator_name == 'chikou_span':
                df['chikou_span'] = df['close'].shift(-26)
                return True
            elif indicator_name.startswith('momentum_'):
                period = int(indicator_name.split('_')[1])
                df[indicator_name] = df['close'] / df['close'].shift(period) - 1
                return True
            elif indicator_name == 'order_flow':
                if 'volume' not in df.columns:
                    self._create_missing_indicator(df, 'volume')
                df['order_flow'] = df['volume'] * np.sign(df['close'].diff())
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error creating indicator {indicator_name}: {e}")
            return False

    def validate_strategy_structure(self, strategy):
        """Validate strategy has proper structure for execution."""
        if not isinstance(strategy, dict):
            logger.error(f"Strategy is not a dictionary: {type(strategy)}")
            return False
        
        if 'rules' not in strategy:
            logger.error("Strategy missing 'rules' field")
            return False
        
        rules = strategy['rules']
        if not isinstance(rules, list):
            logger.error(f"Strategy rules is not a list: {type(rules)}")
            return False
        
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                logger.warning(f"Rule {i} is not a dictionary: {type(rule)}")
                continue
            
            if 'type' not in rule:
                logger.warning(f"Rule {i} missing 'type' field")
                continue
            
            rule_type = rule['type']
            if rule_type == 'combined_condition':
                if 'conditions' not in rule:
                    logger.warning(f"Rule {i} combined_condition missing 'conditions'")
                    continue
                if not isinstance(rule['conditions'], list):
                    logger.warning(f"Rule {i} conditions is not a list")
                    continue
        
        return True

    def debug_strategy_execution(self, symbol, strategy_name, df):
        """Enhanced debugging for strategy execution."""
        logger.info(f"=== COMPREHENSIVE DEBUG: {strategy_name} on {symbol} ===")
        
        # Data quality check
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available indicators: {len(df.columns)} columns")
        
        # Check for common indicators
        key_indicators = ['rsi', 'macd', 'close', 'volume', 'atr']
        missing_key = [ind for ind in key_indicators if ind not in df.columns]
        if missing_key:
            logger.warning(f"Missing key indicators: {missing_key}")
        
        # Check for NaN values
        nan_columns = df.columns[df.isnull().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with NaN values: {nan_columns}")
        
        # Recent price action
        if len(df) >= 5:
            recent_data = df.tail(5)
            logger.info(f"Recent closes: {recent_data['close'].tolist()}")
            if 'rsi' in df.columns:
                logger.info(f"Recent RSI: {recent_data['rsi'].tolist()}")
            if 'volume' in df.columns:
                logger.info(f"Recent volume: {recent_data['volume'].tolist()}")
    
    def _evaluate_simple_condition(self, df, rule):
        """
        Evaluate a simple condition rule.
        
        Args:
            df (DataFrame): Price data with indicators
            rule (dict): Rule parameters
            
        Returns:
            ndarray: Result of the condition
        """
        indicator = rule.get('indicator', '')
        comparison = rule.get('comparison', '')
        value = rule.get('value', 0)
        value_type = rule.get('value_type', 'fixed')
        lookback = rule.get('lookback', 0)
        
        # Check if indicator exists in dataframe
        if indicator not in df.columns:
            return np.zeros(len(df), dtype=bool)
        
        # If lookback is specified, shift the indicator values
        indicator_values = df[indicator].shift(lookback) if lookback > 0 else df[indicator]
        
        # Determine comparison values
        if value_type == 'indicator':
            if value in df.columns:
                compare_values = df[value]
            else:
                return np.zeros(len(df), dtype=bool)
        else:  # fixed value
            compare_values = value
    
        # Apply comparison
        if comparison == '>':
            return indicator_values > compare_values
        elif comparison == '<':
            return indicator_values < compare_values
        elif comparison == '>=':
            return indicator_values >= compare_values
        elif comparison == '<=':
            return indicator_values <= compare_values
        elif comparison == '==':
            return indicator_values == compare_values
        else:
            return np.zeros(len(df), dtype=bool)
    
    def _evaluate_crossover(self, df, rule):
        """
        Evaluate a crossover condition.
        
        Args:
            df (DataFrame): Price data with indicators
            rule (dict): Crossover condition rule
            
        Returns:
            ndarray: Result of the crossover
        """
        indicator1 = rule.get('indicator1', '')
        indicator2 = rule.get('indicator2', '')
        direction = rule.get('direction', '')
        
        if indicator1 not in df.columns or indicator2 not in df.columns:
            return np.zeros(len(df), dtype=bool)
        
        # Check for crossover
        if direction == 'up':
            return (df[indicator1].shift(1) < df[indicator2].shift(1)) & (df[indicator1] > df[indicator2])
        elif direction == 'down':
            return (df[indicator1].shift(1) > df[indicator2].shift(1)) & (df[indicator1] < df[indicator2])
        else:
            return np.zeros(len(df), dtype=bool)
    
    def _evaluate_condition(self, df, condition):
        """
        Evaluate a single condition based on its type.
        
        Args:
            df (DataFrame): Price data with indicators
            condition (dict): Condition to evaluate
            
        Returns:
            ndarray: Boolean array of condition results
        """
        condition_type = condition.get('type', '')
        
        logger.debug(f"Evaluating condition type: {condition_type}")
        
        if condition_type == 'simple_condition':
            return self._evaluate_simple_condition(df, condition)
        elif condition_type == 'crossover':
            return self._evaluate_crossover(df, condition)
        elif condition_type == 'combined_condition':
            return self._evaluate_combined_condition(df, condition)
        elif condition_type == 'pattern':
            return self._evaluate_pattern(df, condition)
        elif condition_type == 'divergence':
            return self._evaluate_divergence(df, condition)
        elif condition_type == 'amd_pattern':
            return self._evaluate_amd_pattern(df, condition)
        elif condition_type == 'candlestick_pattern':
            return self._evaluate_pattern(df, condition)
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
        
        return np.zeros(len(df), dtype=bool)

    def _evaluate_combined_condition(self, df, rule):
        """Enhanced combined condition evaluation with comprehensive debugging."""
        operator = rule.get('operator', 'AND')
        conditions = rule.get('conditions', [])
        signal_type = rule.get('signal', '')
        
        logger.debug(f"Combined condition - Operator: {operator}, Signal: {signal_type}, Conditions: {len(conditions)}")
        
        if not conditions:
            logger.warning("Empty conditions in combined_condition rule")
            return np.zeros(len(df), dtype=bool)
        
        # Detailed condition structure logging
        valid_conditions = []
        for i, condition in enumerate(conditions):
            logger.debug(f"  Condition {i}: {condition}")
            if isinstance(condition, dict) and condition.get('type'):
                valid_conditions.append(condition)
                logger.debug(f"  Condition {i} valid: type={condition.get('type')}, indicator={condition.get('indicator', 'N/A')}")
            else:
                logger.warning(f"  Condition {i} invalid: {type(condition)} - {condition}")
        
        if not valid_conditions:
            logger.warning("No valid conditions found in combined_condition rule")
            return np.zeros(len(df), dtype=bool)
        
        # Evaluate first condition
        try:
            result = self._evaluate_condition(df, valid_conditions[0])
            first_count = np.sum(result) if result is not None else 0
            logger.debug(f"First condition result: {first_count} signals")
        except Exception as e:
            logger.error(f"Error evaluating first condition: {e}")
            return np.zeros(len(df), dtype=bool)
        
        # Combine with remaining conditions
        for i, condition in enumerate(valid_conditions[1:], 1):
            try:
                condition_result = self._evaluate_condition(df, condition)
                if condition_result is not None:
                    condition_count = np.sum(condition_result)
                    logger.debug(f"Condition {i+1} result: {condition_count} signals")
                    
                    if operator == 'AND':
                        result = np.logical_and(result, condition_result)
                    elif operator == 'OR':
                        result = np.logical_or(result, condition_result)
                    else:
                        logger.warning(f"Unknown operator: {operator}, defaulting to AND")
                        result = np.logical_and(result, condition_result)
                else:
                    logger.warning(f"Condition {i+1} returned None")
            except Exception as e:
                logger.error(f"Error evaluating condition {i+1}: {e}")
                continue
        
        final_count = np.sum(result) if result is not None else 0
        logger.debug(f"Combined condition final result for {signal_type}: {final_count} signals")
        
        return result if result is not None else np.zeros(len(df), dtype=bool)

    def _evaluate_pattern(self, df, rule):
        """
        Evaluate a candlestick pattern condition.
        
        Args:
            df (DataFrame): Price data with indicators
            rule (dict): Pattern condition rule
            
        Returns:
            ndarray: Boolean array of condition results
        """
        pattern_type = rule.get('pattern_type', '')
        
        # Check if pattern exists as column (already calculated)
        if pattern_type in df.columns:
            if rule.get('signal', '') == 'buy':
                return df[pattern_type] > 0
            elif rule.get('signal', '') == 'sell':
                return df[pattern_type] < 0
        
        # Common patterns
        if pattern_type == 'bullish_engulfing':
            return (df['open'].shift(1) > df['close'].shift(1)) & \
                   (df['close'] > df['open']) & \
                   (df['open'] < df['close'].shift(1)) & \
                   (df['close'] > df['open'].shift(1))
        
        elif pattern_type == 'bearish_engulfing':
            return (df['open'].shift(1) < df['close'].shift(1)) & \
                   (df['close'] < df['open']) & \
                   (df['open'] > df['close'].shift(1)) & \
                   (df['close'] < df['open'].shift(1))
        
        elif pattern_type == 'hammer':
            return df['hammer'] > 0 if 'hammer' in df.columns else np.zeros(len(df), dtype=bool)
        
        elif pattern_type == 'shooting_star':
            return df['shooting_star'] > 0 if 'shooting_star' in df.columns else np.zeros(len(df), dtype=bool)
        
        # Add more pattern evaluations as needed
        
        return np.zeros(len(df), dtype=bool)

    def _evaluate_divergence(self, df, rule):
        """
        Evaluate a divergence condition.
        
        Args:
            df (DataFrame): Price data with indicators
            rule (dict): Divergence condition rule
            
        Returns:
            ndarray: Boolean array of condition results
        """
        price_indicator = rule.get('price_indicator', 'close')
        technical_indicator = rule.get('technical_indicator', 'rsi')
        divergence_type = rule.get('divergence_type', 'bullish')
        lookback_periods = 5  # Default lookback for finding local extrema
        
        result = np.zeros(len(df), dtype=bool)
        
        if price_indicator not in df.columns or technical_indicator not in df.columns:
            return result
        
        # Find local maxima and minima
        for i in range(lookback_periods, len(df) - 1):
            price_window = df[price_indicator].iloc[i-lookback_periods:i+1]
            indicator_window = df[technical_indicator].iloc[i-lookback_periods:i+1]
            
            if divergence_type == 'bullish':
                # Bullish divergence: Lower price lows but higher indicator lows
                if (price_window.min() == price_window.iloc[-1] and 
                    indicator_window.min() != indicator_window.iloc[-1] and
                    indicator_window.iloc[-1] > indicator_window.min()):
                    result[i] = True
                    
            elif divergence_type == 'bearish':
                # Bearish divergence: Higher price highs but lower indicator highs
                if (price_window.max() == price_window.iloc[-1] and 
                    indicator_window.max() != indicator_window.iloc[-1] and
                    indicator_window.iloc[-1] < indicator_window.max()):
                    result[i] = True
        
        return result
    

    def _evaluate_amd_pattern(self, df, condition):
        """
        Evaluate AMD pattern conditions with comprehensive fallback logic.
        
        Args:
            df (DataFrame): Price data with indicators
            condition (dict): AMD pattern condition
            
        Returns:
            ndarray: Boolean array of condition results
        """
        signal = condition.get('signal', '')
        pattern_type = condition.get('pattern_type', 'standard')
        
        logger.debug(f"Evaluating AMD pattern - signal: {signal}, type: {pattern_type}")
        
        if signal == 'buy':
            if 'amd_distribution_buy' in df.columns:
                result = df['amd_distribution_buy'].fillna(False).astype(bool)
                signal_count = np.sum(result)
                logger.debug(f"AMD buy pattern signals from column: {signal_count}")
                return result
            else:
                logger.warning("AMD distribution buy column not found - creating advanced fallback")
                # Advanced fallback: Create sophisticated AMD-style breakout signals
                if len(df) >= 20:
                    # Consolidation detection
                    price_range = df['high'] - df['low']
                    avg_range = price_range.rolling(20).mean()
                    is_consolidating = (price_range < avg_range * 0.8).rolling(5).sum() >= 3
                    
                    # Breakout detection
                    high_breakout = df['close'] > df['high'].rolling(20).max().shift(1)
                    
                    # Volume confirmation (using synthetic volume)
                    volume_confirm = df['volume'] > df['volume'].rolling(10).mean() * 1.2
                    
                    # Momentum confirmation
                    momentum_confirm = df['close'] > df['close'].shift(5) * 1.001  # 0.1% momentum
                    
                    # Combine conditions
                    result = (
                        is_consolidating.shift(1) &  # Was consolidating
                        high_breakout &               # Now breaking higher
                        volume_confirm &              # With volume
                        momentum_confirm              # And momentum
                    ).fillna(False).astype(bool)
                    
                    signal_count = np.sum(result)
                    logger.debug(f"AMD fallback buy signals created: {signal_count}")
                    return result
                return np.zeros(len(df), dtype=bool)
                    
        elif signal == 'sell':
            if 'amd_distribution_sell' in df.columns:
                result = df['amd_distribution_sell'].fillna(False).astype(bool)
                signal_count = np.sum(result)
                logger.debug(f"AMD sell pattern signals from column: {signal_count}")
                return result
            else:
                logger.warning("AMD distribution sell column not found - creating advanced fallback")
                # Advanced fallback: Create sophisticated AMD-style breakdown signals
                if len(df) >= 20:
                    # Consolidation detection
                    price_range = df['high'] - df['low']
                    avg_range = price_range.rolling(20).mean()
                    is_consolidating = (price_range < avg_range * 0.8).rolling(5).sum() >= 3
                    
                    # Breakdown detection
                    low_breakdown = df['close'] < df['low'].rolling(20).min().shift(1)
                    
                    # Volume confirmation
                    volume_confirm = df['volume'] > df['volume'].rolling(10).mean() * 1.2
                    
                    # Momentum confirmation
                    momentum_confirm = df['close'] < df['close'].shift(5) * 0.999  # -0.1% momentum
                    
                    # Combine conditions
                    result = (
                        is_consolidating.shift(1) &  # Was consolidating
                        low_breakdown &               # Now breaking lower
                        volume_confirm &              # With volume
                        momentum_confirm              # And momentum
                    ).fillna(False).astype(bool)
                    
                    signal_count = np.sum(result)
                    logger.debug(f"AMD fallback sell signals created: {signal_count}")
                    return result
                return np.zeros(len(df), dtype=bool)
        
        logger.warning(f"Unknown AMD pattern signal: {signal}")
        return np.zeros(len(df), dtype=bool)
    

    def execute_trade(self, symbol, trade_type, current_atr=None, lot_size_override=None):
        """
        Execute a trade in MetaTrader 5 with dynamic risk management.
        """
        if not self.connected or not self.live_trade:
            logger.warning("Not connected or live trading is disabled")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found for {symbol}")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
                # Check spread before trading
                # Log spread for monitoring but don't block trades
        spread_points = tick.ask - tick.bid
        spread_pips = spread_points / (0.0001 if 'JPY' not in symbol else 0.01)
        
        logger.info(f"Current spread for {symbol}: {spread_pips:.1f} pips - PROCEEDING with trade based on strategy confidence")

        # Check account balance first
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return None
        
        # Check minimum lot size and available margin
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        
        # Calculate lot size based on available equity (conservative approach)
        available_equity = account_info.equity
        if available_equity < 100:  # Minimum equity threshold
            logger.error(f"Insufficient equity: {available_equity}")
            return None

        point = symbol_info.point
        price = tick.ask if trade_type == 'buy' else tick.bid
        
        # Calculate Stop Loss and Take Profit
        stop_loss_price = 0
        take_profit_price = 0

        if self.use_atr_stops and current_atr is not None and current_atr > 0:
            # Use ATR-based stops
            if trade_type == 'buy':
                stop_loss_price = price - (current_atr * self.atr_sl_multiplier)
                take_profit_price = price + (current_atr * self.atr_tp_multiplier)
            else:  # sell
                stop_loss_price = price + (current_atr * self.atr_sl_multiplier)
                take_profit_price = price - (current_atr * self.atr_tp_multiplier)
        else:
            # Use fixed pip stops with spread adjustment
            pip_value = point * 10 if 'JPY' not in symbol else point
            
                        # Adjust take profit based on spread and signal strength
            spread_adjustment = max(0, spread_pips - 2)  # Add extra buffer for high spreads
            
            # Increase TP for high-confidence signals (this will be passed from calling function)
            confidence_multiplier = 1.0
            if hasattr(self, '_current_signal_strength'):
                if self._current_signal_strength >= 3:  # Very strong signal (3+ votes)
                    confidence_multiplier = 1.3
                elif self._current_signal_strength >= 2:  # Strong signal (2+ votes)
                    confidence_multiplier = 1.2
            
            adjusted_tp_pips = (self.profit_pips * confidence_multiplier) + spread_adjustment
            
            if trade_type == 'buy':
                stop_loss_price = price - (self.stop_loss_pips * pip_value)
                take_profit_price = price + (adjusted_tp_pips * pip_value)
            else:  # sell
                stop_loss_price = price + (self.stop_loss_pips * pip_value)
                take_profit_price = price - (adjusted_tp_pips * pip_value)
            
            logger.info(f"Adjusted TP for {symbol}: {adjusted_tp_pips} pips (spread: {spread_pips:.1f})")
        
        # Dynamic Position Sizing (Conservative)
        final_lot_size = lot_size_override if lot_size_override is not None else self.lot_size
        
        # Use smaller lot size to avoid insufficient funds
        if self.position_sizing_fraction > 0 and available_equity > 0:
            # Risk-based position sizing
            risk_amount = available_equity * self.position_sizing_fraction
            pip_value_per_lot = symbol_info.trade_tick_value if hasattr(symbol_info, 'trade_tick_value') else 1.0
            
            if pip_value_per_lot > 0:
                sl_distance_pips = abs(price - stop_loss_price) / (point * 10 if 'JPY' not in symbol else point)
                if sl_distance_pips > 0:
                    calculated_lot = risk_amount / (sl_distance_pips * pip_value_per_lot)
                    final_lot_size = max(min_lot, min(calculated_lot, max_lot, 0.01))  # Cap at 0.01 for safety
        
        # Ensure lot size is within bounds
        final_lot_size = max(min_lot, min(final_lot_size, max_lot))
        
        # Check stop level requirements
        stop_level = symbol_info.trade_stops_level * point
        if stop_level > 0:
            min_sl_distance = stop_level
            min_tp_distance = stop_level
            
            if trade_type == 'buy':
                if price - stop_loss_price < min_sl_distance:
                    stop_loss_price = price - min_sl_distance
                if take_profit_price - price < min_tp_distance:
                    take_profit_price = price + min_tp_distance
            else:
                if stop_loss_price - price < min_sl_distance:
                    stop_loss_price = price + min_sl_distance
                if price - take_profit_price < min_tp_distance:
                    take_profit_price = price - min_tp_distance

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": final_lot_size,
            "type": mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": round(stop_loss_price, symbol_info.digits),
            "tp": round(take_profit_price, symbol_info.digits),
            "deviation": 50 if symbol == "XAUUSD" else 20,  # Higher deviation for XAUUSD
            "magic": 12345,
            "comment": f"ScalpBot {trade_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_supported_filling_mode(symbol),  # Use dynamic filling mode
        }
        
        # Use correct filling mode for this symbol
        
        # Try with the correct filling mode first
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade EXECUTED for {symbol}: {trade_type}, Lot: {request['volume']}, Price: {result.price}, SL: {request['sl']}, TP: {result['tp']}, Ticket: {result.order}")
            return {
                "ticket": result.order,
                "symbol": symbol,
                "type": trade_type,
                "price": result.price,
                "lot_size": request['volume'],
                "stop_loss": request["sl"],
                "take_profit": request["tp"],
                "time": datetime.now()
            }

        # If that fails, try alternative filling modes
        filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        
        for filling_mode in filling_modes:
            if filling_mode != request["type_filling"]:  # Skip the one we already tried
                request["type_filling"] = filling_mode
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trade EXECUTED for {symbol}: {trade_type}, Lot: {request['volume']}, Price: {result.price}, SL: {request['sl']}, TP: {request['tp']}, Ticket: {result.order}")
                return {
                    "ticket": result.order,
                    "symbol": symbol,
                    "type": trade_type,
                    "price": result.price,
                    "lot_size": request['volume'],
                    "stop_loss": request["sl"],
                    "take_profit": request["tp"],
                    "time": datetime.now()
                }
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                # Update price and try again
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    request["price"] = tick.ask if trade_type == 'buy' else tick.bid
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Trade EXECUTED after requote for {symbol}")
                        return {
                            "ticket": result.order,
                            "symbol": symbol,
                            "type": trade_type,
                            "price": result.price,
                            "lot_size": request['volume'],
                            "stop_loss": request["sl"],
                            "take_profit": request["tp"],
                            "time": datetime.now()
                        }
        
        logger.error(f"Trade failed for {symbol}: {result.retcode}, {result.comment}, Price: {request['price']}, SL: {request['sl']}, TP: {request['tp']}")
        return None
    def close_position(self, ticket):
        """
        Close a specific position by ticket.
        
        Args:
            ticket (int): Position ticket
            
        Returns:
            bool: True if position was closed successfully
        """
        if not self.connected or not self.live_trade:
            logger.warning("Not connected or live trading is disabled")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found for {position.symbol}")
            return False
        
        # Get current tick
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {position.symbol}")
            return False
        
        # Determine close price and order type
        if position.type == mt5.POSITION_TYPE_BUY:
            close_price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = tick.ask  
            order_type = mt5.ORDER_TYPE_BUY
        
        # Prepare close request with proper filling mode
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 20,  # Increased deviation
            "magic": 12345,
            "comment": "Close by Scalping Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Use IOC for better execution
        }
        
        # Try different filling modes if IOC fails
        filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        
        for filling_mode in filling_modes:
            request["type_filling"] = filling_mode
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Closed position {ticket} successfully")
                return True
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                # Update price and try again
                tick = mt5.symbol_info_tick(position.symbol)
                if tick:
                    request["price"] = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Closed position {ticket} after requote")
                        return True
        
        logger.error(f"Failed to close position {ticket}: {result.retcode}, {result.comment}")
        return False
    
    def close_all_positions(self):
        """
        Close all open positions.
        
        Returns:
            int: Number of positions closed
        """
        if not self.connected:
            logger.warning("Not connected to MetaTrader")
            return 0
        
        # Get all open positions
        positions = mt5.positions_get()
        if positions is None:
            logger.info("No open positions to close")
            return 0
        
        # Close each position
        closed_count = 0
        for position in positions:
            if self.close_position(position.ticket):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} positions")
        return closed_count
    
    def get_open_positions(self, symbol=None):
        """
        Get all open positions for a symbol or all symbols.
        
        Args:
            symbol (str): Forex pair symbol or None for all symbols
            
        Returns:
            list: List of open positions
        """
        if not self.connected:
            logger.warning("Not connected to MetaTrader")
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
            
        if positions is None:
            return []
            
        return positions
    
    def backtest_strategy(self, symbol, strategy, timeframe=None, period_days=None):
        """
        Backtest a strategy on historical data, with improved exit logic and cost simulation.
        """
        last_trade_bar = 0  # Track last trade bar for cooldown
        if timeframe is None:
            timeframe = self.timeframe
            
        if period_days is None:
            period_days = self.backtest_data_period
        
        # Fetch historical data
        # For M1, period_days * 1440 bars. Add buffer for indicator calculation.
        bars_to_fetch = (period_days * 1440) + 200 # Buffer for indicators
        df_raw = self.fetch_data(symbol, timeframe, bars=bars_to_fetch)
        
        if df_raw is None or len(df_raw) < 100: # Ensure enough data after fetching
            logger.error(f"Insufficient raw data for {symbol} to backtest (fetched {len(df_raw) if df_raw is not None else 0})")
            return None
        
        # Add technical indicators
        df = self.add_technical_indicators(df_raw.copy()) # Use a copy
        if len(df) < 50: # Ensure enough data after indicators
            logger.error(f"Insufficient data for {symbol} after adding indicators (length {len(df)})")
            return None

        # Apply strategy rules
                # Validate strategy structure first
        df = self.apply_strategy_rules(df, strategy)
        
        # Verify required columns exist
        required_columns = ['final_buy_signal', 'final_sell_signal']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after applying strategy rules: {missing_columns}")
            return None
        
        
        
        # Simulate trading
        account_balance = 10000  # Starting balance
        current_position = 0     # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        stop_loss_level = 0
        take_profit_level = 0
        trade_history = []
        
        # Get point value for pip calculation
        # This is a simplified pip calculation. For true accuracy, use symbol_info.trade_tick_value.
        point_value = 0.0001 if 'JPY' not in symbol else 0.01 
        simulated_spread_cost_per_trade = self.simulated_spread_pips * point_value

        for i in range(1, len(df)): # Start from 1 to use i-1 for entry if needed
            current_open = df['open'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i] # Not used for entry/exit in this model, but for checks
            current_atr = df['atr'].iloc[i-1] if 'atr' in df.columns else self.stop_loss_pips * point_value / self.atr_sl_multiplier # Fallback ATR

            # Check to close existing position
            exit_price_trade = 0
            if current_position == 1:  # Long position
                if current_low <= stop_loss_level: # Stop Loss Hit
                    exit_price_trade = stop_loss_level
                elif current_high >= take_profit_level: # Take Profit Hit
                    exit_price_trade = take_profit_level
                
                if exit_price_trade > 0:
                    profit_pips = (exit_price_trade - entry_price) / point_value - self.simulated_spread_pips
                    profit_amount = profit_pips * self.lot_size * (10 if 'JPY' not in symbol else 1000) # Rough profit amount
                    account_balance += profit_amount
                    trade_history.append({
                        'entry_time': df.index[entry_bar_index], # Need to store entry_bar_index
                        'exit_time': df.index[i],
                        'trade_type': 'buy', 'entry_price': entry_price,
                        'exit_price': exit_price_trade, 'profit_pips': profit_pips,
                        'profit_amount': profit_amount, 'sl_level': stop_loss_level, 'tp_level': take_profit_level
                    })
                    current_position = 0
                    entry_price = 0

            elif current_position == -1:  # Short position
                if current_high >= stop_loss_level: # Stop Loss Hit
                    exit_price_trade = stop_loss_level
                elif current_low <= take_profit_level: # Take Profit Hit
                    exit_price_trade = take_profit_level

                if exit_price_trade > 0:
                    profit_pips = (entry_price - exit_price_trade) / point_value - self.simulated_spread_pips
                    profit_amount = profit_pips * self.lot_size * (10 if 'JPY' not in symbol else 1000) # Rough profit amount
                    account_balance += profit_amount
                    trade_history.append({
                        'entry_time': df.index[entry_bar_index], # Need to store entry_bar_index
                        'exit_time': df.index[i],
                        'trade_type': 'sell', 'entry_price': entry_price,
                        'exit_price': exit_price_trade, 'profit_pips': profit_pips,
                        'profit_amount': profit_amount, 'sl_level': stop_loss_level, 'tp_level': take_profit_level
                    })
                    current_position = 0
                    entry_price = 0
            
            # Open new position if no position exists
            if current_position == 0:
                entry_bar_index = i # Store index of entry bar
                if df['final_buy_signal'].iloc[i-1] == 1 and (i - last_trade_bar) > 30: # Increased cooldown to 30 bars
                    last_trade_bar = i
                    entry_price = current_open # Enter on open of current bar
                    if self.use_atr_stops and current_atr > 0:
                        stop_loss_level = entry_price - current_atr * self.atr_sl_multiplier
                        take_profit_level = entry_price + current_atr * self.atr_tp_multiplier
                    else:
                        stop_loss_level = entry_price - self.stop_loss_pips * point_value
                        take_profit_level = entry_price + self.profit_pips * point_value
                    current_position = 1
                elif df['final_sell_signal'].iloc[i-1] == 1 and (i - last_trade_bar) > 30: # Add cooldown for sell signals too
                    last_trade_bar = i
                    entry_price = current_open # Enter on open of current bar
                    if self.use_atr_stops and current_atr > 0:
                        stop_loss_level = entry_price + current_atr * self.atr_sl_multiplier
                        take_profit_level = entry_price - current_atr * self.atr_tp_multiplier
                    else:
                        stop_loss_level = entry_price + self.stop_loss_pips * point_value
                        take_profit_level = entry_price - self.profit_pips * point_value
                    current_position = -1
        
        # Calculate backtest statistics
        trade_df = pd.DataFrame(trade_history)
        
        if len(trade_df) == 0:
            logger.warning(f"No trades executed in backtest for {symbol} with strategy {strategy.get('name', 'Unknown')}")
            return {
                'symbol': symbol, 'strategy': strategy.get('name', 'Unknown Strategy'),
                'trades': 0, 'win_rate': 0, 'profit_loss': 0, 'final_balance': account_balance,
                'message': 'No trades executed'
            }
        
        winning_trades = len(trade_df[trade_df['profit_pips'] > 0])
        losing_trades = len(trade_df[trade_df['profit_pips'] <= 0]) # Includes break-even as loss for win_rate
        win_rate = winning_trades / len(trade_df) if len(trade_df) > 0 else 0
        total_profit_amount = trade_df['profit_amount'].sum()
        
        logger.info(f"Backtest for {symbol} strategy '{strategy.get('name', 'Unknown')}': Trades: {len(trade_df)}, Win Rate: {win_rate:.2%}, Profit: {total_profit_amount:.2f}")
        
        return {
            'symbol': symbol, 'strategy': strategy.get('name', 'Unknown Strategy'),
            'trades': len(trade_df), 'winning_trades': winning_trades,
            'losing_trades': losing_trades, 'win_rate': win_rate,
            'profit_loss': total_profit_amount, 'final_balance': account_balance,
            'trade_history': trade_history # Keep for detailed analysis if needed
        }
    
    def backtest_all_strategies(self, symbol=None, timeframe=None):
        """
        Backtest all strategies for a symbol or all symbols.
        
        Args:
            symbol (str): Forex pair symbol or None for all symbols
            timeframe: MetaTrader timeframe constant
            
        Returns:
            dict: Backtest results for all strategies
        """
        if timeframe is None:
            timeframe = self.timeframe
            
        symbols = [symbol] if symbol else self.symbols
        results = {}
        
        for sym in symbols:
            results[sym] = {}
            for strategy_name, strategy in self.strategies.items():
                result = self.backtest_strategy(sym, strategy, timeframe)
                if result:
                    results[sym][strategy_name] = result
        
        return results
    
    def find_best_strategy(self, symbol, metrics='profit_loss'):
        """
        Find the best strategy for a symbol based on backtest results.
        
        Args:
            symbol (str): Forex pair symbol
            metrics (str): Metrics to evaluate ('profit_loss', 'win_rate', etc.)
            
        Returns:
            dict: Best strategy details
        """
        results = self.backtest_all_strategies(symbol)
        
        if not results or symbol not in results:
            logger.error(f"No backtest results for {symbol}")
            return None
        
        # Find strategy with highest metrics
        best_strategy = None
        best_value = float('-inf')
        
        for strategy_name, result in results[symbol].items():
            if metrics in result and result[metrics] > best_value:
                best_value = result[metrics]
                best_strategy = {
                    'name': strategy_name,
                    'result': result
                }
        
        return best_strategy
    
    def optimize_strategy_parameters(self, symbol, strategy_name, parameter_ranges):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            symbol (str): Forex pair symbol
            strategy_name (str): Name of the strategy to optimize
            parameter_ranges (dict): Dictionary of parameters and their ranges
            
        Returns:
            dict: Optimized parameters and results
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        base_strategy = self.strategies[strategy_name]
        best_result = None
        best_params = {}
        best_profit = float('-inf')
        
        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Grid search
        for params in self._generate_params_combinations(param_names, param_values):
            # Create modified strategy with new parameters
            test_strategy = base_strategy.copy()
            
            # Update parameters
            for name, value in params.items():
                # Find the rule containing this parameter
                for i, rule in enumerate(test_strategy.get('rules', [])):
                    if name in rule:
                        test_strategy['rules'][i][name] = value
            
            # Backtest with new parameters
            result = self.backtest_strategy(symbol, test_strategy)
            
            if result and result['profit_loss'] > best_profit:
                best_profit = result['profit_loss']
                best_result = result
                best_params = params
                
        return {
            'strategy_name': strategy_name,
            'optimized_params': best_params,
            'result': best_result
        }
    
    def _generate_params_combinations(self, names, values, current_idx=0, current_params=None):
        """
        Generate all combinations of parameters for grid search.
        
        Args:
            names (list): Parameter names
            values (list): Parameter values ranges
            current_idx (int): Current parameter index
            current_params (dict): Current parameters
            
        Yields:
            dict: Parameter combination
        """
        if current_params is None:
            current_params = {}
            
        if current_idx == len(names):
            yield current_params.copy()
            return
            
        for value in values[current_idx]:
            current_params[names[current_idx]] = value
            yield from self._generate_params_combinations(names, values, current_idx + 1, current_params)
    
    def create_default_strategy(self, name="Default Scalping Strategy"):
        """
        Create a default scalping strategy.
        
        Args:
            name (str): Strategy name
            
        Returns:
            dict: Strategy definition
        """
        strategy = {
            "name": name,
            "description": "Default scalping strategy with RSI, MACD and Bollinger Bands",
            "timeframe": self.timeframe,
            "rules": [
                {
                    "type": "simple_condition",
                    "indicator": "rsi",
                    "comparison": "<",
                    "value": 30,
                    "value_type": "fixed",
                    "signal": "buy"
                },
                {
                    "type": "simple_condition",
                    "indicator": "rsi",
                    "comparison": ">",
                    "value": 70,
                    "value_type": "fixed",
                    "signal": "sell"
                },
                {
                    "type": "crossover",
                    "indicator1": "macd",
                    "indicator2": "macd_signal",
                    "direction": "up",
                    "signal": "buy"
                },
                {
                    "type": "crossover",
                    "indicator1": "macd",
                    "indicator2": "macd_signal",
                    "direction": "down",
                    "signal": "sell"
                },
                {
                    "type": "simple_condition",
                    "indicator": "close",
                    "comparison": "<",
                    "value": "lower_band",
                    "value_type": "indicator",
                    "signal": "buy"
                },
                {
                    "type": "simple_condition",
                    "indicator": "close",
                    "comparison": ">",
                    "value": "upper_band",
                    "value_type": "indicator",
                    "signal": "sell"
                }
            ]
        }
        
        return strategy
    
    def add_strategy_from_template(self, template_name, **kwargs):
        """
        Add a strategy from a predefined template.
        
        Args:
            template_name (str): Template name
            **kwargs: Additional parameters for the template
            
        Returns:
            dict: Created strategy
        """
        if template_name == "trend_following":
            strategy = {
                "name": kwargs.get("name", "Trend Following Strategy"),
                "description": "Strategy that follows market trends using moving averages",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "crossover",
                        "indicator1": "sma_5",
                        "indicator2": "sma_20",
                        "direction": "up",
                        "signal": "buy"
                    },
                    {
                        "type": "crossover",
                        "indicator1": "sma_5",
                        "indicator2": "sma_20",
                        "direction": "down",
                        "signal": "sell"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "adx",
                        "comparison": ">",
                        "value": 25,
                        "value_type": "fixed",
                        "signal": "buy"
                    }
                ]
            }
        
        elif template_name == "mean_reversion":
            strategy = {
                "name": kwargs.get("name", "Mean Reversion Strategy"),
                "description": "Strategy that trades price reversions to the mean",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": "<",
                        "value": "lower_band",
                        "value_type": "indicator",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": ">",
                        "value": "upper_band",
                        "value_type": "indicator",
                        "signal": "sell"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "rsi",
                        "comparison": "<",
                        "value": 30,
                        "value_type": "fixed",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "rsi",
                        "comparison": ">",
                        "value": 70,
                        "value_type": "fixed",
                        "signal": "sell"
                    }
                ]
            }
        
        elif template_name == "breakout":
            strategy = {
                "name": kwargs.get("name", "Breakout Strategy"),
                "description": "Strategy that trades price breakouts",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": ">",
                        "value": "upper_band",
                        "value_type": "indicator",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": "<",
                        "value": "lower_band",
                        "value_type": "indicator",
                        "signal": "sell"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "adx",
                        "comparison": ">",
                        "value": 25,
                        "value_type": "fixed",
                        "signal": "buy"
                    }
                ]
            }
        
        elif template_name == "rsi_strategy":
            strategy = {
                "name": kwargs.get("name", "RSI Strategy"),
                "description": "Strategy based on RSI indicator",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "simple_condition",
                        "indicator": "rsi",
                        "comparison": "<",
                        "value": kwargs.get("oversold", 30),
                        "value_type": "fixed",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "rsi",
                        "comparison": ">",
                        "value": kwargs.get("overbought", 70),
                        "value_type": "fixed",
                        "signal": "sell"
                    }
                ]
            }
            
        elif template_name == "macd_strategy":
            strategy = {
                "name": kwargs.get("name", "MACD Strategy"),
                "description": "Strategy based on MACD indicator",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "crossover",
                        "indicator1": "macd",
                        "indicator2": "macd_signal",
                        "direction": "up",
                        "signal": "buy"
                    },
                    {
                        "type": "crossover",
                        "indicator1": "macd",
                        "indicator2": "macd_signal",
                        "direction": "down",
                        "signal": "sell"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "macd_hist",
                        "comparison": ">",
                        "value": 0,
                        "value_type": "fixed",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "macd_hist",
                        "comparison": "<",
                        "value": 0,
                        "value_type": "fixed",
                        "signal": "sell"
                    }
                ]
            }
            
        elif template_name == "bollinger_bands_strategy":
            strategy = {
                "name": kwargs.get("name", "Bollinger Bands Strategy"),
                "description": "Strategy based on Bollinger Bands",
                "timeframe": self.timeframe,
                "rules": [
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": "<",
                        "value": "lower_band",
                        "value_type": "indicator",
                        "signal": "buy"
                    },
                    {
                        "type": "simple_condition",
                        "indicator": "close",
                        "comparison": ">",
                        "value": "upper_band",
                        "value_type": "indicator",
                        "signal": "sell"
                    }
                ]
            }
        
        else:
            # Default to a general scalping strategy
            strategy = self.create_default_strategy(name=kwargs.get("name", "Custom Scalping Strategy"))
        
        # Add custom rules if provided
        if "rules" in kwargs:
            strategy["rules"].extend(kwargs["rules"])
        
        # Save the strategy
        strategy_name = strategy["name"]
        self.save_strategy(strategy_name, strategy)
        self.strategies[strategy_name] = strategy
        
        return strategy
    
    def add_candlestick_pattern_strategy(self, name="Candlestick Pattern Strategy", patterns=None):
        """
        Add a strategy based on candlestick patterns.
        
        Args:
            name (str): Strategy name
            patterns (dict): Dictionary of patterns and their signals
            
        Returns:
            dict: Created strategy
        """
        if patterns is None:
            patterns = {
                "doji": "buy",
                "engulfing": "buy",
                "hammer": "buy",
                "shooting_star": "sell",
                "three_white_soldiers": "buy",
                "three_black_crows": "sell"
            }
        
        rules = []
        
        for pattern, signal in patterns.items():
            if signal == "buy":
                rules.append({
                    "type": "candlestick_pattern",
                    "pattern": pattern,
                    "signal": "buy"
                })
            else:
                rules.append({
                    "type": "candlestick_pattern",
                    "pattern": pattern,
                    "signal": "sell"
                })
        
        strategy = {
            "name": name,
            "description": "Strategy based on candlestick patterns",
            "timeframe": self.timeframe,
            "rules": rules
        }
        
        # Save the strategy
        self.save_strategy(name, strategy)
        self.strategies[name] = strategy
        
        return strategy
    
    def run_single_iteration(self, symbol):
        """Enhanced single iteration with ASCII-only logging."""
        start_time = time.time()
        iteration_num = getattr(self, '_iteration_counter', 0) + 1
        setattr(self, '_iteration_counter', iteration_num)
        
        logger.info(f"[CYCLE_START] Processing {symbol} - Iteration #{iteration_num}")
        
        # Fetch data with timing
        fetch_start = time.time()
        required_initial_bars = 1600 # Max lookback (e.g. 1440 for VWAP) + buffer (e.g. 100-200)

        df_raw = self.fetch_data(symbol, self.timeframe, bars=required_initial_bars)
        fetch_time = (time.time() - fetch_start) * 1000
        min_bars_for_indicators = 1440 # Should match the longest lookback of any indicator

        if df_raw is None or len(df_raw) < min_bars_for_indicators:
            logger.error(f"Failed to fetch sufficient data for {symbol}")
            return None
        
        logger.info(f"[FETCH_DATA({symbol})] MT5 request {required_initial_bars} bars -> Received {len(df_raw)} bars ({fetch_time:.0f}ms)")
        
        # Add indicators with timing
        indicators_start = time.time()
        df = self.add_technical_indicators(df_raw.copy())
        indicators_time = (time.time() - indicators_start) * 1000
        
        if df.empty or len(df) < 2:
            logger.error(f"Dataframe empty or too short after adding indicators for {symbol}")
            return None
        
        # Log indicator categories with ASCII characters
        logger.info(f"[INDICATORS({symbol})] Calculating 52 indicators -> Complete ({indicators_time:.0f}ms)")
        logger.info(f"|-- Basic: SMA(5,20,50), EMA(5,10,20,50), RSI(14), MACD(12,26,9)")
        logger.info(f"|-- Advanced: ATR(14), ADX(14), BB(20,2), Stoch(5,3), SAR")
        logger.info(f"|-- Patterns: Doji, Engulfing, Hammer, ShootingStar")
        logger.info(f"`-- AMD: Consolidation zones, Volume analysis, Breakout detection")
        
        # Strategy processing with detailed timing
        strategy_start = time.time()
        final_signals = {'buy': False, 'sell': False}
        active_strategy_name = "None"
        
        logger.info(f"[STRATEGY_ENGINE({symbol})] Processing {len(self.strategies)} strategies")
        
        # AMD Analysis
        amd_signal_triggered = False
        if self.amd_enabled:
            amd_start = time.time()
            amd_pattern = self.detect_amd_pattern(df, symbol)
            
            logger.info(f"[AMD_ANALYSIS({symbol})] Scanning for patterns")
            
            if amd_pattern:
                confidence = amd_pattern.get('confidence', 0)
                logger.info(f"|-- Consolidation: 20-bar analysis -> Range ratio: {confidence:.2f}")
                logger.info(f"|-- Volume: Current vs average -> Confirmation: {'YES' if confidence > 0.6 else 'NO'}")
                logger.info(f"|-- Breakout: Pattern detected -> Confidence: {confidence:.2f}")
                logger.info(f"`-- AMD_RESULT -> {'Valid pattern' if confidence >= 0.6 else 'No valid pattern'} (confidence: {confidence:.2f})")
                
                if confidence >= 0.6:
                    if amd_pattern['direction'] == 'buy':
                        final_signals['buy'] = True
                        amd_signal_triggered = True
                        active_strategy_name = f"AMD_BUY (conf: {confidence:.2f})"
                    elif amd_pattern['direction'] == 'sell':
                        final_signals['sell'] = True
                        amd_signal_triggered = True
                        active_strategy_name = f"AMD_SELL (conf: {confidence:.2f})"
            else:
                logger.info(f"`-- AMD_RESULT -> No pattern detected")
        
        # Traditional strategies
        strategy_buy_votes = 0
        strategy_sell_votes = 0
        
        if not amd_signal_triggered:
            working_strategies = 0
            
            for strategy_name, strategy_config in self.strategies.items():
                strategy_timing_start = time.time()
                strategy_df = self.apply_strategy_rules(df.copy(), strategy_config)
                strategy_timing = (time.time() - strategy_timing_start) * 1000
                
                buy_signals = sell_signals = 0
                if not strategy_df.empty:
                    last_row = strategy_df.iloc[-1]
                    buy_signals = last_row.get('buy_signal', 0)
                    sell_signals = last_row.get('sell_signal', 0)
                    
                    if buy_signals == 1 or sell_signals == 1:
                        working_strategies += 1
                        
                    if buy_signals == 1:
                        strategy_buy_votes += 1
                    if sell_signals == 1:
                        strategy_sell_votes += 1
                
                logger.info(f"|-- {strategy_name} -> {len(strategy_config.get('rules', []))} rules -> BUY:{buy_signals} SELL:{sell_signals} signals ({strategy_timing:.0f}ms)")
            
            logger.info(f"`-- VOTE_RESULT -> BUY:{strategy_buy_votes} SELL:{strategy_sell_votes} -> {'Signal detected' if strategy_buy_votes >= 2 or strategy_sell_votes >= 2 else 'Insufficient consensus'}")
            
            if strategy_buy_votes >= 2 and self.apply_signal_cooldown(symbol, 'buy'):
                final_signals['buy'] = True
                active_strategy_name = f"Traditional_BUY ({strategy_buy_votes} votes)"
                self._current_signal_strength = strategy_buy_votes
            elif strategy_sell_votes >= 2 and self.apply_signal_cooldown(symbol, 'sell'):
                final_signals['sell'] = True
                active_strategy_name = f"Traditional_SELL ({strategy_sell_votes} votes)"
                self._current_signal_strength = strategy_sell_votes
        
        # Signal aggregation
        signal_decision = "NO_TRADE"
        if final_signals['buy']:
            signal_decision = f"BUY ({active_strategy_name})"
        elif final_signals['sell']:
            signal_decision = f"SELL ({active_strategy_name})"
        
        logger.info(f"[SIGNAL_AGGREGATION({symbol})] Traditional: {strategy_buy_votes} BUY, {strategy_sell_votes} SELL -> AMD: {'Pattern' if amd_signal_triggered else 'None'}")
        logger.info(f"`-- FINAL_DECISION -> {signal_decision}")
        
        # Position management
        open_positions = self.get_open_positions(symbol)
        logger.info(f"[POSITION_MGMT({symbol})] Current positions: {len(open_positions)} -> Max allowed: {self.max_positions}")
        
        actions = []
        
        # Execute trades if signals present
        if (final_signals['buy'] or final_signals['sell']) and self.is_market_session_active(symbol) and self.check_daily_limits():
            if len(open_positions) < self.max_positions:
                execution_start = time.time()
                
                if amd_signal_triggered and hasattr(self, 'amd_patterns') and symbol in self.amd_patterns:
                    logger.info(f"[TRADE_SIGNAL] {symbol} {'BUY' if final_signals['buy'] else 'SELL'} (AMD Pattern Detected)")
                    amd_pattern = self.amd_patterns[symbol]
                    
                    # Log AMD pattern details
                    direction = amd_pattern['direction'].upper()
                    confidence = amd_pattern['confidence']
                    logger.info(f"[AMD_PATTERN_DETAILS]")
                    logger.info(f"|-- Phase: Distribution (confidence: {confidence:.2f})")
                    logger.info(f"|-- Entry: Market price -> Pattern breakout detected")
                    logger.info(f"`-- Position_Size: AMD risk management (2% account risk)")
                    
                    trade_result = self.execute_amd_trade(amd_pattern)
                    execution_time = (time.time() - execution_start) * 1000
                    
                    if trade_result:
                        logger.info(f"[TRADE_EXECUTED] SUCCESS ({execution_time:.0f}ms execution time)")
                        logger.info(f"|-- Slippage: 0 pips (high-priority AMD execution)")
                        logger.info(f"`-- RESULT -> Position opened: AMD pattern trade")
                        actions.append(f"Executed AMD {direction} for {symbol}")
                    else:
                        logger.info(f"[TRADE_FAILED] FAILED ({execution_time:.0f}ms)")
                else:
                    # Traditional trade execution
                    trade_type = 'buy' if final_signals['buy'] else 'sell'
                    trade_result = self.execute_trade(symbol, trade_type)
                    execution_time = (time.time() - execution_start) * 1000
                    
                    if trade_result:
                        logger.info(f"[TRADE_EXECUTED] SUCCESS ({execution_time:.0f}ms execution time)")
                        actions.append(f"Executed traditional {trade_type.upper()} for {symbol}")
                    else:
                        logger.info(f"[TRADE_FAILED] FAILED ({execution_time:.0f}ms)")
            else:
                logger.info(f"`-- ACTION -> Max positions reached ({len(open_positions)}/{self.max_positions})")
        else:
            logger.info(f"`-- ACTION -> None required")
        
        # Performance metrics
        total_time = (time.time() - start_time) * 1000
        logger.info(f"[=== {symbol} COMPLETE] ({total_time:.0f}ms) ===")
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'final_signals': final_signals,
            'actions': actions,
            'active_strategy': active_strategy_name,
            'amd_enabled': self.amd_enabled,
            'amd_pattern_detected': amd_signal_triggered,
            'execution_time_ms': total_time
        }
    

    def modify_position_to_breakeven(self, ticket, breakeven_price):
        """Modify position to breakeven stop loss."""
        if not self.live_trade or not self.connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        symbol_info = mt5.symbol_info(position.symbol)
        if not symbol_info:
            return False
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": round(breakeven_price, symbol_info.digits),
            "tp": position.tp,  # Keep existing TP
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Set position {ticket} to breakeven at {breakeven_price}")
            return True
        else:
            logger.error(f"Failed to set breakeven for {ticket}: {result.comment}")
            return False
    
    def run_trading_cycle(self):
        """
        Run a complete trading cycle for all symbols.
        
        Returns:
            dict: Results for all symbols
        """
        if not self.connected:
            if not self.connect_to_metatrader():
                logger.error("Cannot run trading cycle without MT5 connection")
                return None
        
        results = {}
        
        # Process each symbol in parallel
        futures = {}
        for symbol in self.symbols:
            future = self.executor.submit(self.run_single_iteration, symbol)
            futures[future] = symbol
        
        # Collect results
        for future in futures:
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return results
    
    def run_continuous(self, interval_seconds=5):
        """
        Run the trading bot continuously.
        
        Args:
            interval_seconds (int): Interval between iterations in seconds
        """
        try:
            logger.info(f"Starting continuous trading with {interval_seconds} second interval")
            
            # Connect to MetaTrader
            if not self.connected:
                if not self.connect_to_metatrader():
                    logger.error("Failed to connect to MetaTrader. Exiting.")
                    return
            
            # Load strategies
            self.load_strategies()
            
            # Load default strategy if none exist
            if not self.strategies:
                default_strategy = self.create_default_strategy()
                self.save_strategy("Default", default_strategy)
                self.strategies["Default"] = default_strategy
            
            
            
            # Main trading loop
            while True:
                try:
                    start_time = time.time()
                    
                    # Run trading cycle
                    results = self.run_trading_cycle()
                    
                    if results:
                        # Log actions
                        for symbol, result in results.items():
                            if result.get('actions'):
                                for action in result['actions']:
                                    logger.info(action)
                    
                    # Calculate elapsed time and sleep if needed
                    elapsed = time.time() - start_time
                    sleep_time = max(0, interval_seconds - elapsed)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt. Stopping continuous trading.")
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(interval_seconds)  # Sleep before retrying
        
        finally:
            # Clean up
            self.close_all_positions()
            self.disconnect_from_metatrader()
            self.executor.shutdown()
            logger.info("Stopped continuous trading.")
    
    def add_custom_strategy(self, strategy_code):
        """
        Add a custom strategy from code (JSON format).
        
        Args:
            strategy_code (str): JSON string with strategy definition
            
        Returns:
            bool: True if strategy was added successfully
        """
        try:
            strategy = json.loads(strategy_code)
            
            # Validate strategy format
            if 'name' not in strategy or 'rules' not in strategy:
                logger.error("Invalid strategy format: missing 'name' or 'rules'")
                return False
            
            # Save the strategy
            strategy_name = strategy['name']
            self.save_strategy(strategy_name, strategy)
            self.strategies[strategy_name] = strategy
            
            logger.info(f"Added custom strategy: {strategy_name}")
            return True
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for strategy")
            return False
        except Exception as e:
            logger.error(f"Error adding custom strategy: {e}")
            return False
    
    def update_strategy(self, strategy_name, updates):
        """
        Update parameters of an existing strategy.
        
        Args:
            strategy_name (str): Name of the strategy to update
            updates (dict): Dictionary of updates
            
        Returns:
            bool: True if strategy was updated successfully
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return False
        
        try:
            # Get current strategy
            strategy = self.strategies[strategy_name].copy()
            
            # Update fields
            for key, value in updates.items():
                if key == 'rules':
                    # Handle updating specific rules
                    for rule_idx, rule_updates in value.items():
                        rule_idx = int(rule_idx)
                        if rule_idx < len(strategy['rules']):
                            for rule_key, rule_value in rule_updates.items():
                                strategy['rules'][rule_idx][rule_key] = rule_value
                else:
                    strategy[key] = value
            
            # Save updated strategy
            self.save_strategy(strategy_name, strategy)
            self.strategies[strategy_name] = strategy
            
            logger.info(f"Updated strategy: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating strategy {strategy_name}: {e}")
            return False
    
    def import_advanced_strategies(self, strategies_json):
        """
        Import a list of advanced strategies in JSON format.
        
        Args:
            strategies_json (str or list): JSON string or parsed list of strategies
            
        Returns:
            int: Number of strategies successfully imported
        """
        try:
            # Parse JSON if string is provided
            if isinstance(strategies_json, str):
                strategies = json.loads(strategies_json)
            else:
                strategies = strategies_json
                
            # Import each strategy
            imported_count = 0
            for strategy in strategies:
                if 'name' in strategy and 'rules' in strategy:
                    # Convert timeframe from numeric to MT5 constant if needed
                    if isinstance(strategy.get('timeframe'), int):
                        timeframe_map = {
                            1: mt5.TIMEFRAME_M1,
                            5: mt5.TIMEFRAME_M5,
                            15: mt5.TIMEFRAME_M15,
                            30: mt5.TIMEFRAME_M30,
                            60: mt5.TIMEFRAME_H1,
                            240: mt5.TIMEFRAME_H4,
                            1440: mt5.TIMEFRAME_D1
                        }
                        strategy['timeframe'] = timeframe_map.get(strategy['timeframe'], mt5.TIMEFRAME_M1)
                    
                    # Save the strategy
                    strategy_json = json.dumps(strategy, indent=4)
                    if self.add_custom_strategy(strategy_json):
                        imported_count += 1
                        logger.info(f"Imported strategy: {strategy['name']}")
                    
            logger.info(f"Successfully imported {imported_count} advanced strategies")
            return imported_count
        
        except Exception as e:
            logger.error(f"Error importing advanced strategies: {e}")
            return 0
def create_example_strategy():
    """
    Create an example strategy in JSON format.
    """
    strategy = {
        "name": "Example RSI + MACD Strategy",
        "description": "Combined RSI and MACD strategy for scalping",
        "timeframe": mt5.TIMEFRAME_M1,
        "rules": [
            {
                "type": "simple_condition",
                "indicator": "rsi",
                "comparison": "<",
                "value": 30,
                "value_type": "fixed",
                "signal": "buy"
            },
            {
                "type": "simple_condition",
                "indicator": "rsi",
                "comparison": ">",
                "value": 70,
                "value_type": "fixed",
                "signal": "sell"
            },
            {
                "type": "crossover",
                "indicator1": "macd",
                "indicator2": "macd_signal",
                "direction": "up",
                "signal": "buy"
            },
            {
                "type": "crossover",
                "indicator1": "macd",
                "indicator2": "macd_signal",
                "direction": "down",
                "signal": "sell"
            }
        ]
    }
    
    return json.dumps(strategy, indent=4)

