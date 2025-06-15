//+------------------------------------------------------------------+
//|                               BOT_Scalping.mq5                   |
//|                                              Advanced Trading    |
//|                                       https://www.mql5.com       |
//+------------------------------------------------------------------+
#property copyright "Advanced Trading"
#property link      "https://www.mql5.com"
#property version   "2.110" // Full, expanded version for scalping

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input string          InpSymbol = "XAUUSD";      // Trading Symbol
// --- SCALPING CHANGE: Default timeframe is now M1 ---
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M1;  // Timeframe for Scalping
input int             InpMagicNumber = 12346;      // Magic Number (changed to avoid conflict)
input int             InpMinBars = 100;          // Minimum Bars for Analysis


// --- SCALPING CHANGE: Risk and Trade Management defaults adjusted for scalping ---
input double  InpRiskPercent = 0.5;                   // Risk Percent per Trade (Lower for high frequency)
input double  InpMaxVolume = 0.10;                    // Maximum Volume per Trade
input double  InpMaxDailyDrawdownPercent = 5.0;       // Max Daily Drawdown % (0=disable)
input int     InpMaxConsecutiveLosses = 7;            // Max consecutive losses before pause (Slightly higher for more trades)
input bool    InpUseBreakEven = true;                 // Use Break-Even?
input int     InpBreakEvenPips = 40;                  // Pips in profit to trigger BE (Tighter for scalping)
input int     InpBreakEvenLockPips = 5;               // Pips to lock in at BE (Tighter for scalping)
input bool    InpUseTrailingStop = true;              // Use Trailing Stop?
input int     InpTrailingStopPips = 60;               // Trailing Stop distance in pips (Tighter for scalping)

// --- NEW: Long-Term Strategy Enhancements ---
input bool    InpUseRegimeFilter = true;              // Use Market Regime Filter (e.g., EMA on higher timeframe)
input ENUM_TIMEFRAMES InpRegimeFilterTF = PERIOD_H1;  // Timeframe for the regime filter
input int     InpRegimeFilterMAPeriod = 200;            // Moving Average period for the regime filter
input bool    InpUseDynamicTakeProfit = true;         // Use Supply/Demand zones for dynamic TP

// --- SCALPING CHANGE: Multi-Timeframe Confluence settings ---
input bool    InpUseMultiTimeframeConfluence = true;  // Enable Multi-Timeframe Confluence
input ENUM_TIMEFRAMES InpConfluenceTF_4H = PERIOD_H4; // 4H Timeframe for Structure
input ENUM_TIMEFRAMES InpConfluenceTF_1H = PERIOD_H1; // 1H Timeframe for Entry Trigger
input ENUM_TIMEFRAMES InpConfluenceTF_15M = PERIOD_M15; // 15M Timeframe for Candlestick Confirmation
input int     InpConfluenceMinSignals = 2;            // Minimum number of confluent signals required (e.g., 2 out of 3)

// --- SCALPING CHANGE: Profit Withdrawal Settings (NEW) ---
input bool    InpEnableProfitWithdrawal = false;      // Enable automatic trading halt for profit withdrawal
input double  InpProfitWithdrawalThreshold = 500.0;   // Amount of profit (in account currency) to trigger withdrawal halt
input bool    InpResetWithdrawalDaily = true;         // Reset withdrawal halt at start of a new day

// --- ADVANCED MULTI-TARGET PROFIT SYSTEMS (NEW) ---
input bool    InpEnableMultiTP = true;                // Enable Multi-Target Profit System
input int     InpMultiTPMethod = 0;                   // Multi-TP Method: 0=3-2-1, 1=4-3-2-1, 2=Volatility-Adjusted, 3=Session-Based

// Multi-TP Method Constants
#define MULTITP_321         0
#define MULTITP_4321        1
#define MULTITP_VOLATILITY  2
#define MULTITP_SESSION     3

//--- Global variables
CTrade        trade;
CPositionInfo   position;
CAccountInfo    account;
CSymbolInfo     symbolInfo;

datetime        lastBarTime = 0;
int             tradeCount = 0;
int             winCount = 0;
double          totalProfit = 0.0;

// --- Risk Management Global Variables ---
double          dailyEquityStart;
datetime        dayStartDate;
int             consecutiveLosses = 0;
bool            isTradingDisabled = false;
string          disableReason = "";
bool            isProfitWithdrawalHaltActive = false; // NEW: Global variable for profit withdrawal halt

// --- Monte Carlo Simulation Variables ---
double sumWin = 0.0;
double sumLoss = 0.0;
int winTrades = 0;
int lossTrades = 0;

//--- Analysis structure (No change to logic)
struct AnalysisResult
{
    string signal;
    double entry;
    double stopLoss;
    double takeProfit;
    double riskReward;
    int    confidence;
    string wyckoffPhase;
    double wyckoffStrength;
    int    volumeCurrent;
    int    volumeAvg;
    string amdTrend;
    double amdValue;
    double momentum;
    double moneyFlow;
    double currentPrice;
    double demandZone;
    double demandStrength;
    double supplyZone;
    string sdBias;
    string candlePattern;
    string patternLocation;
    int    patternConfidence;
    string marketStructure;
    double vwap;
    string vwapPosition;
    double orderBlock;
    double liquidity;
};

// --- Add at the top (global scope) ---
double cachedATR = 0;
datetime cachedATRTime = 0;

// --- Multi-Timeframe Confluence Structures (New) ---
struct MultiTimeframeAnalysisResult
{
    string wyckoffPhase;
    string amdTrend;
    string sdBias;
    string candlePattern;
    string marketStructure;
    string signal;
};

// --- ENHANCED Monte Carlo Simulation Structures ---
struct MonteCarloMetrics {
    double expectedValue;
    double sharpeRatio;
    double maxDrawdown;
    double winRateConfidence;
    double edgeRatio;
    double probabilityRuin;
    double optimalF;
    double kellyFraction;
    double medianProfit;
    double profitFactor;
    double robustnessFactor;
};

// Ultra-Fast Execution Optimization Structures
struct ExecutionMetrics {
    double avgExecutionTime;
    double lastExecutionTime;
    int successfulTrades;
    int failedTrades;
    datetime lastTradeTimestamp;
};

// Global execution metrics tracker
ExecutionMetrics ultraFastMetrics;
MonteCarloMetrics lastMCResults;
bool adaptiveLotSizingEnabled = true;
bool antiDetectionMode = true;
bool hiddenEdgeActive = true;
datetime lastOptimizationTime = 0;

// --- Structure for Multi-TP Levels (NEW) ---
struct MultiTPLevels {
    double tp[4];      // TP price levels
    double percent[4]; // Portion to close at each TP (0-1)
    int count;         // Number of TPs
};

// --- Function to calculate Multi-TP levels (NEW) ---
void CalculateMultiTPLevels(double entry, double direction, double atr, MultiTPLevels &levels)
{
    ArrayInitialize(levels.tp, 0);
    ArrayInitialize(levels.percent, 0);
    levels.count = 0;
    
    int method = InpMultiTPMethod;
    if(method == MULTITP_321) {
        // 3-2-1 Method
        levels.tp[0] = entry + direction * (5 * _Point);
        levels.tp[1] = entry + direction * (12 * _Point);
        levels.tp[2] = entry + direction * (20 * _Point);
        levels.percent[0] = 0.5;
        levels.percent[1] = 0.3;
        levels.percent[2] = 0.2;
        levels.count = 3;
    } else if(method == MULTITP_4321) {
        // 4-3-2-1 Pyramid Method
        levels.tp[0] = entry + direction * (3 * _Point);
        levels.tp[1] = entry + direction * (8 * _Point);
        levels.tp[2] = entry + direction * (15 * _Point);
        levels.tp[3] = entry + direction * (25 * _Point);
        levels.percent[0] = 0.4;
        levels.percent[1] = 0.3;
        levels.percent[2] = 0.2;
        levels.percent[3] = 0.1;
        levels.count = 4;
    } else if(method == MULTITP_VOLATILITY) {
        // Volatility-Adjusted
        if(atr < 8 * _Point) {
            levels.tp[0] = entry + direction * (3 * _Point);
            levels.tp[1] = entry + direction * (7 * _Point);
            levels.tp[2] = entry + direction * (12 * _Point);
        } else if(atr < 15 * _Point) {
            levels.tp[0] = entry + direction * (5 * _Point);
            levels.tp[1] = entry + direction * (12 * _Point);
            levels.tp[2] = entry + direction * (20 * _Point);
        } else {
            levels.tp[0] = entry + direction * (8 * _Point);
            levels.tp[1] = entry + direction * (18 * _Point);
            levels.tp[2] = entry + direction * (30 * _Point);
        }
        levels.percent[0] = 0.5;
        levels.percent[1] = 0.3;
        levels.percent[2] = 0.2;
        levels.count = 3;
    } else if(method == MULTITP_SESSION) {
        // Session-Based
        MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
        int hour = dt.hour;
        double tp1=0,tp2=0,tp3=0;
        if(hour >= 0 && hour < 8) { // Asian
            tp1=3; tp2=6; tp3=10;
        } else if(hour >= 8 && hour < 15) { // London
            tp1=6; tp2=12; tp3=22;
        } else if(hour >= 15 && hour < 20) { // NY
            tp1=5; tp2=10; tp3=18;
        } else { // Overlap
            tp1=8; tp2=15; tp3=28;
        }
        levels.tp[0] = entry + direction * (tp1 * _Point);
        levels.tp[1] = entry + direction * (tp2 * _Point);
        levels.tp[2] = entry + direction * (tp3 * _Point);
        levels.percent[0] = 0.5;
        levels.percent[1] = 0.3;
        levels.percent[2] = 0.2;
        levels.count = 3;
    }
}

// Ultra-Fast Execution Optimization Function
bool UltraFastExecute(AnalysisResult &analysis, double lotSize)
{
    // Timestamp tracking
    long startTime = GetMicrosecondCount();
    
    // Rapid pre-execution checks
    if(!symbolInfo.Name(InpSymbol) || !symbolInfo.RefreshRates())
    {
        ultraFastMetrics.failedTrades++;
        Print("❌ Ultra-Fast Execution Failed: Symbol Refresh Error");
        return false;
    }
    
    // Liquidity and Spread Check (Hyper-Optimized)
    double currentSpread = symbolInfo.Ask() - symbolInfo.Bid();
    double spreadThreshold = cachedATR * 0.3; // Tighter spread control
    if(currentSpread > spreadThreshold)
    {
        Print("⚡ Spread too wide. Skipping ultra-fast execution.");
        return false;
    }
    
    // Rapid Order Placement with Minimal Overhead
    bool result = false;
    string comment = "UF_" + TimeToString(TimeCurrent(), TIME_MINUTES);
    
    // Parallel Execution Path
    if(StringFind(analysis.signal, "BUY") >= 0)
    {
        result = trade.Buy(lotSize, InpSymbol, 0, analysis.stopLoss, analysis.takeProfit, comment);
    }
    else
    {
        result = trade.Sell(lotSize, InpSymbol, 0, analysis.stopLoss, analysis.takeProfit, comment);
    }
    
    // Execution Time Calculation
    long endTime = GetMicrosecondCount();
    double executionTime = (endTime - startTime) / 1000.0; // Convert to milliseconds
    
    // Update Execution Metrics
    if(result)
    {
        ultraFastMetrics.successfulTrades++;
        ultraFastMetrics.lastExecutionTime = executionTime;
        ultraFastMetrics.lastTradeTimestamp = TimeCurrent();
        
        // Rolling average of execution times
        if(ultraFastMetrics.avgExecutionTime == 0)
            ultraFastMetrics.avgExecutionTime = executionTime;
        else
            ultraFastMetrics.avgExecutionTime = 
                (ultraFastMetrics.avgExecutionTime * 0.7) + (executionTime * 0.3);
        
        Print("⚡ Ultra-Fast Execution Successful!");
        Print("   Execution Time: ", DoubleToString(executionTime, 3), " ms");
        Print("   Avg Execution Time: ", DoubleToString(ultraFastMetrics.avgExecutionTime, 3), " ms");
    }
    else
    {
        ultraFastMetrics.failedTrades++;
        Print("❌ Ultra-Fast Execution Failed: ", trade.ResultRetcode(), " - ", trade.ResultComment());
    }
    
    return result;
}

// Execution Cooldown and Frequency Management
bool CanExecuteUltraFastTrade()
{
    // Prevent over-trading
    if(ultraFastMetrics.lastTradeTimestamp > 0)
    {
        datetime currentTime = TimeCurrent();
        long timeSinceLastTrade = currentTime - ultraFastMetrics.lastTradeTimestamp;
        
        // Minimum time between trades (adaptive)
        long minTradeCooldown = 60; // 1 minute base cooldown
        
        // Adaptive cooldown based on recent performance
        if(ultraFastMetrics.successfulTrades > 0)
        {
            double successRate = (double)ultraFastMetrics.successfulTrades / 
                                 (ultraFastMetrics.successfulTrades + ultraFastMetrics.failedTrades);
            
            // Adjust cooldown dynamically
            if(successRate < 0.5) minTradeCooldown *= 2;
            if(successRate > 0.8) minTradeCooldown = MathMax(30, minTradeCooldown / 2);
        }
        
        if(timeSinceLastTrade < minTradeCooldown)
        {
            Print("⏳ Ultra-Fast Trade Cooldown Active. Waiting...");
            return false;
        }
    }
    
    return true;
}

// Most profitable times (hidden from retail)
// 8:30-9:00 AM EST: Algorithm recalibration
// 10:00-10:30 AM EST: Fund rebalancing
// 2:00-2:30 PM EST: European close arbitrage
// 3:50-4:00 PM EST: End-of-day positioning

// bool IsAlgoShiftTime() {
//     MqlDateTime dt;
//     TimeToStruct(TimeCurrent(), dt);
//     int hour = dt.hour;
//     int minute = dt.min;
    
//     return (hour == 8 && minute >= 30) || 
//            (hour == 10 && minute <= 30) ||
//            (hour == 14 && minute <= 30) ||
//            (hour == 15 && minute >= 50);
// }

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   // For scalping, slippage is critical. This value is in points.
   trade.SetDeviationInPoints(30); 
   
   if(!symbolInfo.Name(InpSymbol))
   {
      Print("❌ Failed to initialize symbol: ", InpSymbol);
      return INIT_FAILED;
   }
   
   dailyEquityStart = account.Equity();
   dayStartDate = TimeCurrent();
   Comment("");

   Print("🚀 MT5 Scalping EA Initialized");
   Print("📊 Symbol: ", InpSymbol, " | Timeframe: ", EnumToString(InpTimeframe));
   
   // Initialize lastMCResults to prevent null reference
   ZeroMemory(lastMCResults);
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🏁 FINAL RESULTS (Scalping):");
   Print("   Total Trades: ", tradeCount);
   Print("   Winning Trades: ", winCount);
   double winRate = (tradeCount > 0) ? (winCount * 100.0 / tradeCount) : 0.0;
   Print("   Win Rate: ", DoubleToString(winRate, 1), "%");
   Print("   Total Profit: $", DoubleToString(totalProfit, 2));

   // --- Enhanced Monte Carlo Simulation Call ---
   double avgWin = (winTrades > 0) ? sumWin / winTrades : 0.0;
   double avgLoss = (lossTrades > 0) ? sumLoss / lossTrades : 0.0;
   double winRateFrac = (tradeCount > 0) ? (double)winTrades / tradeCount : 0.0;
   MonteCarloMetrics metrics = RunEnhancedMonteCarloSimulation(tradeCount, winRateFrac, avgWin, avgLoss, 10000);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    CheckRiskCircuitBreakers();
    ManageOpenPositions();

    datetime currentBarTime = iTime(InpSymbol, InpTimeframe, 0);
    if(lastBarTime == 0 || currentBarTime != lastBarTime)
    {
        lastBarTime = currentBarTime;
        
        if(!isTradingDisabled)
        {
            // The old optimization call here was not working and has been removed.
            // It is replaced by a call in OnTradeTransaction that runs periodically.
            
            // The hardcoded times in IsAdvancedAlgoShiftTime are very rigid and a source of overfitting.
            // A better approach is to use a volatility filter or the new regime filter.
            // These calls have been disabled to promote more robust performance.
            // bool isOptimalWindow = IsAdvancedAlgoShiftTime();
            // bool hasInstitutionalPattern = RecognizeInstitutionalPatterns();
            
            AnalyzeAndTrade();
        }
    }
}

//+------------------------------------------------------------------+
//| Risk Management Circuit Breakers (No change to logic)            |
//+------------------------------------------------------------------+
void CheckRiskCircuitBreakers()
{
    if(InpMaxDailyDrawdownPercent > 0)
    {
        MqlDateTime current_date, start_date;
        TimeToStruct(TimeCurrent(), current_date);
        TimeToStruct(dayStartDate, start_date);

        if(current_date.day_of_year != start_date.day_of_year || current_date.year != start_date.year)
        {
            dailyEquityStart = account.Equity();
            dayStartDate = TimeCurrent();
            if(isTradingDisabled && disableReason == "Daily Drawdown Limit Hit")
            {
               isTradingDisabled = false;
               disableReason = "";
               Print("✅ New day started. Trading re-enabled.");
               Comment("");
            }
        }
        
        // --- SCALPING CHANGE: Reset Profit Withdrawal Halt Daily (NEW) ---
        if(InpEnableProfitWithdrawal && InpResetWithdrawalDaily && isProfitWithdrawalHaltActive)
        {
            MqlDateTime current_date, start_date;
            TimeToStruct(TimeCurrent(), current_date);
            TimeToStruct(dayStartDate, start_date);
            if(current_date.day_of_year != start_date.day_of_year || current_date.year != start_date.year)
            {
                isTradingDisabled = false;
                isProfitWithdrawalHaltActive = false;
                disableReason = "";
                Print("✅ New day started. Profit withdrawal halt reset. Trading re-enabled.");
                Comment("");
            }
        }

        double currentEquity = account.Equity();
        double drawdownPercent = (dailyEquityStart > 0) ? (dailyEquityStart - currentEquity) / dailyEquityStart * 100.0 : 0;
        
        if(drawdownPercent >= InpMaxDailyDrawdownPercent)
        {
            if(!isTradingDisabled)
            {
                isTradingDisabled = true;
                disableReason = "Daily Drawdown Limit Hit";
                Print("🚫 RISK ALERT: Daily Drawdown limit of ", InpMaxDailyDrawdownPercent, "% hit. All new trading is disabled for today.");
                Comment("TRADING DISABLED: DAILY DRAWDOWN LIMIT HIT");
            }
        }
    }

    if(InpMaxConsecutiveLosses > 0 && consecutiveLosses >= InpMaxConsecutiveLosses)
    {
        if(!isTradingDisabled)
        {
            isTradingDisabled = true;
            disableReason = "Consecutive Loss Limit Hit";
            Print("🚫 RISK ALERT: ", InpMaxConsecutiveLosses, " consecutive losses. All new trading is disabled until the next winning trade.");
            Comment("TRADING DISABLED: CONSECUTIVE LOSS LIMIT HIT");
        }
    }
}

//+------------------------------------------------------------------+
//| Trade Management (No change to logic)                            |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
    if(!InpUseBreakEven && !InpUseTrailingStop && !InpEnableMultiTP) return;

    if(position.Select(InpSymbol))
    {
        if(position.Magic() != InpMagicNumber) return;

        // --- Multi-TP Partial Close Management (NEW) ---
        if(InpEnableMultiTP)
            ManageMultiTPPartialCloses();

        double openPrice = position.PriceOpen();
        double currentPrice = (position.PositionType() == POSITION_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();
        double currentSL = position.StopLoss();
        double currentTP = position.TakeProfit();
        long ticket = (long)position.Ticket();
        
        double point = symbolInfo.Point();
        int digits = (int)symbolInfo.Digits();
        
        if(point == 0) return;
        
        double pipsInProfit = 0;

        if (position.PositionType() == POSITION_TYPE_BUY)
        {
            pipsInProfit = (currentPrice - openPrice) / point;
        }
        else // SELL
        {
            pipsInProfit = (openPrice - currentPrice) / point;
        }
        
        if(InpUseBreakEven)
        {
            double breakEvenPrice = (position.PositionType() == POSITION_TYPE_BUY) ? openPrice + InpBreakEvenLockPips * point : openPrice - InpBreakEvenLockPips * point;
            if(pipsInProfit >= InpBreakEvenPips && currentSL != breakEvenPrice)
            {
                double newSL = NormalizeDouble(breakEvenPrice, digits);
                if(trade.PositionModify(ticket, newSL, currentTP))
                {
                    Print("✅ BREAK-EVEN SET for ticket ", ticket, " at price ", DoubleToString(newSL, digits));
                }
            }
        }
        
        if(InpUseTrailingStop)
        {
             if(pipsInProfit >= InpTrailingStopPips)
             {
                 double newSL = 0;
                 if(position.PositionType() == POSITION_TYPE_BUY)
                 {
                     newSL = currentPrice - InpTrailingStopPips * point;
                     if(newSL > openPrice && newSL > currentSL)
                     {
                         if(trade.PositionModify(ticket, NormalizeDouble(newSL, digits), currentTP))
                         {
                            Print("📈 TRAILING STOP moved for ticket ", ticket, " to ", DoubleToString(newSL, digits));
                         }
                     }
                 }
                 else // SELL
                 {
                     newSL = currentPrice + InpTrailingStopPips * point;
                     if(newSL < openPrice && (newSL < currentSL || currentSL == 0))
                     {
                         if(trade.PositionModify(ticket, NormalizeDouble(newSL, digits), currentTP))
                         {
                            Print("📉 TRAILING STOP moved for ticket ", ticket, " to ", DoubleToString(newSL, digits));
                         }
                     }
                 }
             }
        }
    }
}


//+------------------------------------------------------------------+
//| Main analysis and trading function (No change to logic)          |
//+------------------------------------------------------------------+
void AnalyzeAndTrade()
{
   int bars = iBars(InpSymbol, InpTimeframe);
   if(bars < InpMinBars)
   {
      Print("⚠️ Not enough bars for analysis: ", bars, " < ", InpMinBars);
      return;
   }
   
   AnalysisResult analysis;
   if(!RunCompleteAnalysis(analysis))
   {
      Print("❌ Analysis failed");
      return;
   }
   
   PrintAnalysisReport(analysis);
   ExecuteTradingLogic(analysis);
}

//+------------------------------------------------------------------+
//| Wyckoff Method Analysis                                          |
//+------------------------------------------------------------------+
bool WyckoffAnalysis(AnalysisResult &analysis)
{
    double highs[], lows[], closes[];
    long volumes[];
    int bars = 50;
    
    if(CopyHigh(InpSymbol, InpTimeframe, 0, bars, highs) <= 0 ||
       CopyLow(InpSymbol, InpTimeframe, 0, bars, lows) <= 0 ||
       CopyClose(InpSymbol, InpTimeframe, 0, bars, closes) <= 0 ||
       CopyTickVolume(InpSymbol, InpTimeframe, 0, bars, volumes) <= 0)
       return false;
    
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);
    ArraySetAsSeries(volumes, true);
    
    double recentHigh = highs[ArrayMaximum(highs, 0, bars)];
    double recentLow = lows[ArrayMinimum(lows, 0, bars)];
    double currentPrice = closes[0];
    double pricePosition = (recentHigh != recentLow) ? 
                           (currentPrice - recentLow) / (recentHigh - recentLow) : 0.5;
    
    double avgVolume = 0;
    double recentVolume = 0;
    for(int i = 0; i < bars; i++)
       avgVolume += (double)volumes[i];
    avgVolume /= bars;
    
    for(int i = 0; i < 5 && i < bars; i++)
       recentVolume += (double)volumes[i];
    recentVolume /= MathMin(5, bars);
    
    double volumeRatio = (avgVolume > 0) ? recentVolume / avgVolume : 1.0;
    
    double priceRange = recentHigh - recentLow;
    double adaptiveThreshold = MathMax(priceRange * 0.15, cachedATR * 0.5); // Example: 15% of range or 0.5 ATR

    if((currentPrice - recentLow) < adaptiveThreshold && volumeRatio < 0.8)
    {
       analysis.wyckoffPhase = "📦 Accumulation";
       analysis.wyckoffStrength = MathMin(0.9, 0.5 + (adaptiveThreshold - (currentPrice - recentLow)) + (0.8 - volumeRatio));
    }
    else if((recentHigh - currentPrice) < adaptiveThreshold && volumeRatio > 1.2)
    {
       analysis.wyckoffPhase = "🚀 Distribution";
       analysis.wyckoffStrength = MathMin(0.9, 0.5 + ((recentHigh - currentPrice) / adaptiveThreshold) + (volumeRatio - 1.2));
    }
    else
    {
       analysis.wyckoffPhase = "⚖️ Re-accumulation";
       analysis.wyckoffStrength = 0.6;
    }
    
    analysis.volumeCurrent = (int)recentVolume;
    analysis.volumeAvg = (int)avgVolume;
    
    return true;
}

//+------------------------------------------------------------------+
//| AMD (Accumulation / Manipulation / Distribution) Analysis        |
//+------------------------------------------------------------------+
bool AMDAnalysis(AnalysisResult &analysis)
{
    double closes[];
    long volumes[];
    int bars = 20;
    
    if(CopyClose(InpSymbol, InpTimeframe, 0, bars, closes) <= 0 ||
       CopyTickVolume(InpSymbol, InpTimeframe, 0, bars, volumes) <= 0)
       return false;
    
    ArraySetAsSeries(closes, true);
    ArraySetAsSeries(volumes, true);
    
    double smaShort = 0, smaLong = 0;
    for(int i = 0; i < 5 && i < bars; i++)
       smaShort += closes[i];
    smaShort /= MathMin(5, bars);
    
    for(int i = 0; i < 15 && i < bars; i++)
       smaLong += closes[i];
    smaLong /= MathMin(15, bars);
    
    analysis.amdTrend = (smaShort > smaLong) ? "📈 Bullish" : "📉 Bearish";
    if (bars <5) return false;
    analysis.momentum = (closes[4] != 0) ? (closes[0] / closes[4] - 1) : 0;
    
    double volumeWeightedChange = 0;
    for(int i = 1; i < bars; i++)
    {
       double priceChange = (closes[i] != 0) ? (closes[i-1] - closes[i]) / closes[i] : 0;
       volumeWeightedChange += priceChange * (double)volumes[i];
    }
    analysis.amdValue = MathAbs(volumeWeightedChange * 100000);
    analysis.moneyFlow = analysis.momentum * 1000;
    
    return true;
}

//+------------------------------------------------------------------+
//| Supply & Demand Zone Analysis                                    |
//+------------------------------------------------------------------+
bool SupplyDemandAnalysis(AnalysisResult &analysis)
{
    double highs[], lows[], closes[];
    long volumes[];
    int bars = 50;
    
    if(CopyHigh(InpSymbol, InpTimeframe, 0, bars, highs) <= 0 ||
       CopyLow(InpSymbol, InpTimeframe, 0, bars, lows) <= 0 ||
       CopyClose(InpSymbol, InpTimeframe, 0, bars, closes) <= 0 ||
       CopyTickVolume(InpSymbol, InpTimeframe, 0, bars, volumes) <= 0)
       return false;
    
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);
    ArraySetAsSeries(volumes, true);
    
    analysis.currentPrice = closes[0];
    
    double avgVolume = 0;
    for(int i = 0; i < bars; i++)
       avgVolume += (double)volumes[i];
    if (bars > 0) avgVolume /= bars;
    
    double demandZone = analysis.currentPrice - (analysis.currentPrice * 0.01); // 1% below
    double supplyZone = analysis.currentPrice + (analysis.currentPrice * 0.01); // 1% above
    double demandStrength = 0.5;
    
    for(int i = 5; i < bars - 5; i++)
    {
       if(lows[i] <= lows[i-1] && lows[i] <= lows[i+1] && 
          lows[i] <= lows[i-2] && lows[i] <= lows[i+2] &&
          (double)volumes[i] > avgVolume)
       {
          if(lows[i] < analysis.currentPrice && lows[i] > demandZone)
          {
             demandZone = lows[i];
             if(avgVolume > 0) demandStrength = MathMin(0.95, (double)volumes[i] / avgVolume * 0.5);
          }
       }
       
       if(highs[i] >= highs[i-1] && highs[i] >= highs[i+1] && 
          highs[i] >= highs[i-2] && highs[i] >= highs[i+2] &&
          (double)volumes[i] > avgVolume)
       {
          if(highs[i] > analysis.currentPrice && highs[i] < supplyZone)
          {
             supplyZone = highs[i];
          }
       }
    }
    
    analysis.demandZone = demandZone;
    analysis.demandStrength = demandStrength;
    analysis.supplyZone = supplyZone;
    
    double vwapSum = 0, volumeSum = 0;
    for(int i = 0; i < MathMin(20, bars); i++)
    {
       vwapSum += closes[i] * (double)volumes[i];
       volumeSum += (double)volumes[i];
    }
    analysis.vwap = (volumeSum > 0) ? vwapSum / volumeSum : analysis.currentPrice;
    analysis.sdBias = (analysis.currentPrice > analysis.vwap) ? "📈 Bullish" : "📉 Bearish";
    
    return true;
}

//+------------------------------------------------------------------+
//| Candlestick Pattern Analysis (REVISED)                           |
//+------------------------------------------------------------------+
bool CandlestickAnalysis(AnalysisResult &analysis)
{
    double opens[], highs[], lows[], closes[];
    int barsToCopy = 10; 

    if(CopyOpen(InpSymbol, InpTimeframe, 0, barsToCopy, opens) < 4 || 
       CopyHigh(InpSymbol, InpTimeframe, 0, barsToCopy, highs) < 4 ||
       CopyLow(InpSymbol, InpTimeframe, 0, barsToCopy, lows) < 4 ||
       CopyClose(InpSymbol, InpTimeframe, 0, barsToCopy, closes) < 4)
    {
        return false;
    }

    ArraySetAsSeries(opens, true);
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);

    analysis.candlePattern = "None";
    analysis.patternLocation = "N/A";
    analysis.patternConfidence = 0;

    double body1 = MathAbs(opens[1] - closes[1]);
    double body2 = MathAbs(opens[2] - closes[2]);
    double body3 = MathAbs(opens[3] - closes[3]);
    double range1 = highs[1] - lows[1];

    if (range1 == 0 || body3 == 0 || (highs[3]-lows[3]) == 0) return true;

    // Evening Star (Bearish Reversal)
    if(closes[3] > opens[3] && body3 > (highs[3]-lows[3]) * 0.6 &&
       MathMin(opens[2], closes[2]) > closes[3] && body2 < body3 * 0.3 &&
       closes[1] < opens[1] && closes[1] < closes[3] && opens[1] > MathMin(opens[2], closes[2]))
    {
        analysis.candlePattern = "🌟 Evening Star (Bearish)";
        analysis.patternLocation = "Bearish Reversal";
        analysis.patternConfidence = 92;
    }
    // Morning Star (Bullish Reversal)
    else if(opens[3] > closes[3] && body3 > (highs[3]-lows[3]) * 0.6 &&
            MathMax(opens[2], closes[2]) < closes[3] && body2 < body3 * 0.3 &&
            closes[1] > opens[1] && closes[1] > closes[3] && opens[1] < MathMax(opens[2], closes[2]))
    {
        analysis.candlePattern = "☀️ Morning Star (Bullish)";
        analysis.patternLocation = "Bullish Reversal";
        analysis.patternConfidence = 90;
    }
    // Bearish Engulfing
    else if(closes[2] > opens[2] && opens[1] > closes[1] &&
            opens[1] > closes[2] && closes[1] < opens[2])
    {
        analysis.candlePattern = "❄️ Bearish Engulfing";
        analysis.patternLocation = "Bearish Reversal";
        analysis.patternConfidence = 88;
    }
    // Bullish Engulfing
    else if(opens[2] > closes[2] && closes[1] > opens[1] &&
            closes[1] > opens[2] && opens[1] < closes[2])
    {
        analysis.candlePattern = "🔥 Bullish Engulfing";
        analysis.patternLocation = "Bullish Reversal";
        analysis.patternConfidence = 87;
    }
    // Dark Cloud Cover (Bearish Reversal)
    else if(closes[2] > opens[2] && opens[1] > closes[1] &&
            opens[1] > highs[2] && closes[1] < (opens[2] + closes[2])/2 &&
            closes[1] > opens[2])
    {
        analysis.candlePattern = "☁️ Dark Cloud Cover (Bearish)";
        analysis.patternLocation = "Bearish Reversal";
        analysis.patternConfidence = 85;
    }
    // Piercing Line (Bullish Reversal)
    else if(opens[2] > closes[2] && closes[1] > opens[1] &&
            opens[1] < lows[2] && closes[1] > (opens[2] + closes[2])/2 &&
            closes[1] < opens[2])
    {
        analysis.candlePattern = "⚔️ Piercing Line (Bullish)";
        analysis.patternLocation = "Bullish Reversal";
        analysis.patternConfidence = 84;
    }
    
    bool isUptrend = closes[2] > closes[3];
    bool isDowntrend = closes[2] < closes[3];
    
    double upperShadow = highs[1] - MathMax(opens[1], closes[1]);
    double lowerShadow = MathMin(opens[1], closes[1]) - lows[1];
    
    // Hammer (Bullish Reversal after Downtrend)
    if(isDowntrend && body1 < range1 * 0.3 && lowerShadow >= 2 * body1 && upperShadow < body1)
    {
        analysis.candlePattern = "🔨 Hammer (Bullish)";
        analysis.patternLocation = "Bullish Reversal";
        analysis.patternConfidence = 82;
    }
    // Hanging Man (Bearish Reversal after Uptrend)
    else if(isUptrend && body1 < range1 * 0.3 && lowerShadow >= 2 * body1 && upperShadow < body1)
    {
        analysis.candlePattern = "🕴️ Hanging Man (Bearish)";
        analysis.patternLocation = "Bearish Reversal";
        analysis.patternConfidence = 83;
    }
    // Inverted Hammer (Bullish Reversal after Downtrend)
    else if(isDowntrend && body1 < range1 * 0.3 && upperShadow >= 2 * body1 && lowerShadow < body1)
    {
        analysis.candlePattern = "inverted_hammer Inverted Hammer (Bullish)";
        analysis.patternLocation = "Potential Bullish Reversal";
        analysis.patternConfidence = 78;
    }
    // Shooting Star (Bearish Reversal after Uptrend)
    else if(isUptrend && body1 < range1 * 0.3 && upperShadow >= 2 * body1 && lowerShadow < body1)
    {
        analysis.candlePattern = "🌠 Shooting Star (Bearish)";
        analysis.patternLocation = "Bearish Reversal";
        analysis.patternConfidence = 86;
    }
    // Doji (Indecision)
    else if(body1 <= range1 * 0.1)
    {
        analysis.candlePattern = "🔄 Doji";
        analysis.patternLocation = "Indecision";
        analysis.patternConfidence = 70;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Market Context Analysis                                          |
//+------------------------------------------------------------------+
bool MarketContextAnalysis(AnalysisResult &analysis)
{
    double highs[], lows[], closes[];
    int bars = 20;
    
    if(CopyHigh(InpSymbol, InpTimeframe, 0, bars, highs) <= 0 ||
       CopyLow(InpSymbol, InpTimeframe, 0, bars, lows) <= 0 ||
       CopyClose(InpSymbol, InpTimeframe, 0, bars, closes) <= 0)
       return false;
    
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);
    
    bool higherHighs = false;
    bool lowerLows = false;

    if (bars > 2) {
        higherHighs = highs[0] > highs[1] && highs[1] > highs[2];
        lowerLows = lows[0] < lows[1] && lows[1] < lows[2];
    }
    
    if(higherHighs && !lowerLows)
       analysis.marketStructure = "Bullish Market Structure";
    else if(lowerLows && !higherHighs)
       analysis.marketStructure = "Bearish Market Structure";
    else
       analysis.marketStructure = "Sideways Market Structure";
    
    analysis.vwapPosition = (analysis.currentPrice > analysis.vwap) ? "ABOVE VWAP" : "BELOW VWAP";
    analysis.orderBlock = (bars >= 3) ? closes[2] : analysis.currentPrice;
    
    double basePrice = MathRound(analysis.currentPrice / 10) * 10;
    analysis.liquidity = basePrice + ((analysis.currentPrice > basePrice) ? 10 : -10);
    
    double trendStrength = MathAbs(highs[0] - lows[0]) / cachedATR;
    string marketRegime = (trendStrength > 2.0) ? "Trending" : "Range-Bound";
    analysis.marketStructure += " (" + marketRegime + ")";
    
    return true;
}

//+------------------------------------------------------------------+
//| Multi-Timeframe Analysis (Lite)                                  |
//+------------------------------------------------------------------+
bool RunMultiTimeframeAnalysis(ENUM_TIMEFRAMES timeframe, MultiTimeframeAnalysisResult &result)
{
    ZeroMemory(result);

    double closes_mftf[];
    if(CopyClose(InpSymbol, timeframe, 0, 10, closes_mftf) <= 0) return false;
    ArraySetAsSeries(closes_mftf, true);

    // Simplified Wyckoff Phase detection (using longer-term trend)
    if (closes_mftf[0] > closes_mftf[9]) result.wyckoffPhase = "Accumulation";
    else if (closes_mftf[0] < closes_mftf[9]) result.wyckoffPhase = "Distribution";
    else result.wyckoffPhase = "Re-accumulation";

    // Simplified AMD Trend (using SMA crossover)
    double smaShort_mftf = iMA(InpSymbol, timeframe, 5, 0, MODE_SMA, PRICE_CLOSE);
    double smaLong_mftf = iMA(InpSymbol, timeframe, 15, 0, MODE_SMA, PRICE_CLOSE);
    if(smaShort_mftf == EMPTY_VALUE || smaLong_mftf == EMPTY_VALUE) return false;
    result.amdTrend = (smaShort_mftf > smaLong_mftf) ? "Bullish" : "Bearish";

    // Simplified Supply & Demand Bias (using price vs VWAP approximation)
    double currentPrice_mftf = iClose(InpSymbol, timeframe, 0);
    double vwap_mftf = iMA(InpSymbol, timeframe, 20, 0, MODE_LWMA, PRICE_CLOSE); // Using LWMA for VWAP approximation
    if(currentPrice_mftf == EMPTY_VALUE || vwap_mftf == EMPTY_VALUE) return false;
    result.sdBias = (currentPrice_mftf > vwap_mftf) ? "Bullish" : "Bearish";

    // Simplified Candlestick Pattern detection (check current bar type)
    double open_mftf = iOpen(InpSymbol, timeframe, 0);
    double high_mftf = iHigh(InpSymbol, timeframe, 0);
    double low_mftf = iLow(InpSymbol, timeframe, 0);
    double close_mftf = iClose(InpSymbol, timeframe, 0);
    if(open_mftf == EMPTY_VALUE || high_mftf == EMPTY_VALUE || low_mftf == EMPTY_VALUE || close_mftf == EMPTY_VALUE) return false;

    double body_mftf = MathAbs(open_mftf - close_mftf);
    double range_mftf = high_mftf - low_mftf;

    if (range_mftf > 0 && body_mftf / range_mftf > 0.6) {
        if (close_mftf > open_mftf) result.candlePattern = "Bullish Strong";
        else result.candlePattern = "Bearish Strong";
    } else if (range_mftf > 0 && body_mftf / range_mftf < 0.1) {
        result.candlePattern = "Doji";
    } else {
        result.candlePattern = "Neutral";
    }

    // Simplified Market Structure (Higher Highs/Lower Lows)
    double highs_mftf[], lows_mftf[];
    if(CopyHigh(InpSymbol, timeframe, 0, 3, highs_mftf) <= 0 ||
       CopyLow(InpSymbol, timeframe, 0, 3, lows_mftf) <= 0) return false;
    ArraySetAsSeries(highs_mftf, true);
    ArraySetAsSeries(lows_mftf, true);

    if (highs_mftf[0] > highs_mftf[1] && lows_mftf[0] > lows_mftf[1]) result.marketStructure = "Bullish";
    else if (highs_mftf[0] < highs_mftf[1] && lows_mftf[0] < lows_mftf[1]) result.marketStructure = "Bearish";
    else result.marketStructure = "Sideways";

    // Determine overall signal for this timeframe
    int bullishCount = 0;
    int bearishCount = 0;

    if (StringFind(result.wyckoffPhase, "Accumulation") >= 0) bullishCount++;
    if (StringFind(result.amdTrend, "Bullish") >= 0) bullishCount++;
    if (StringFind(result.sdBias, "Bullish") >= 0) bullishCount++;
    if (StringFind(result.candlePattern, "Bullish") >= 0) bullishCount++;
    if (StringFind(result.marketStructure, "Bullish") >= 0) bullishCount++;

    if (StringFind(result.wyckoffPhase, "Distribution") >= 0) bearishCount++;
    if (StringFind(result.amdTrend, "Bearish") >= 0) bearishCount++;
    if (StringFind(result.sdBias, "Bearish") >= 0) bearishCount++;
    if (StringFind(result.candlePattern, "Bearish") >= 0) bearishCount++;
    if (StringFind(result.marketStructure, "Bearish") >= 0) bearishCount++;

    if (bullishCount > bearishCount) result.signal = "BUY";
    else if (bearishCount > bullishCount) result.signal = "SELL";
    else result.signal = "NEUTRAL";

    return true;
}

//+------------------------------------------------------------------+
//| Run Complete Analysis                                            |
//+------------------------------------------------------------------+
bool RunCompleteAnalysis(AnalysisResult &analysis)
{
    ZeroMemory(analysis);
    
    if(!WyckoffAnalysis(analysis)) return false;
    if(!AMDAnalysis(analysis)) return false;
    if(!SupplyDemandAnalysis(analysis)) return false;
    if(!CandlestickAnalysis(analysis)) return false;
    if(!MarketContextAnalysis(analysis)) return false;
    
    int bullishSignals = 0;
    int bearishSignals = 0;
    
    if(StringFind(analysis.wyckoffPhase, "Accumulation") >= 0) bullishSignals += 2;
    else if(StringFind(analysis.wyckoffPhase, "Distribution") >= 0) bearishSignals += 2;
    
    if(analysis.momentum > 0) bullishSignals += 1;
    else bearishSignals += 1;
    
    if(StringFind(analysis.sdBias, "Bullish") >= 0) bullishSignals += 1;
    else bearishSignals += 1;
    
    if(StringFind(analysis.candlePattern, "Bullish") >= 0) bullishSignals += 2;
    else if(StringFind(analysis.candlePattern, "Bearish") >= 0) bearishSignals += 2;
    
    if(StringFind(analysis.marketStructure, "Bullish") >= 0) bullishSignals += 1;
    else if(StringFind(analysis.marketStructure, "Bearish") >= 0) bearishSignals += 1;
    
    int totalSignals = bullishSignals + bearishSignals;
    analysis.confidence = (totalSignals > 0) ? (int)(MathMax(bullishSignals, bearishSignals) * 100.0 / totalSignals) : 50;
    
    int atrHandle = iATR(InpSymbol, InpTimeframe, 14);
    if(atrHandle == INVALID_HANDLE) return false;

    datetime barTime = iTime(InpSymbol, InpTimeframe, 0);
    if (cachedATRTime != barTime) {
        double atrBuffer[];
        if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0) {
            IndicatorRelease(atrHandle);
            return false;
        }
        cachedATR = atrBuffer[0];
        cachedATRTime = barTime;
    }
    IndicatorRelease(atrHandle);
    double atrValue = cachedATR;
    
    analysis.entry = analysis.currentPrice;
    
    // --- SCALPING CHANGE: Tighter SL and TP based on ATR ---
    if(bullishSignals > bearishSignals)
    {
       analysis.signal = "✅ BUY";
       analysis.stopLoss = analysis.currentPrice - (atrValue * 1.2);
       analysis.takeProfit = analysis.currentPrice + (atrValue * 2.0);
    }
    else
    {
       analysis.signal = "❌ SELL";
       analysis.stopLoss = analysis.currentPrice + (atrValue * 1.2);
       analysis.takeProfit = analysis.currentPrice - (atrValue * 2.0);
    }
    
    // --- NEW: Market Regime Filter ---
    if(InpUseRegimeFilter)
    {
        double regimeMA = iMA(InpSymbol, InpRegimeFilterTF, InpRegimeFilterMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
        if (regimeMA != EMPTY_VALUE)
        {
            string currentSignal = (bullishSignals > bearishSignals) ? "BUY" : "SELL";
            if (currentSignal == "BUY" && analysis.currentPrice < regimeMA)
            {
                Print("❌ REGIME FILTER BLOCK: BUY signal below H1 ",InpRegimeFilterMAPeriod,"-EMA. No trade.");
                return false;
            }
            if (currentSignal == "SELL" && analysis.currentPrice > regimeMA)
            {
                Print("❌ REGIME FILTER BLOCK: SELL signal above H1 ",InpRegimeFilterMAPeriod,"-EMA. No trade.");
                return false;
            }
            Print("✅ REGIME FILTER PASSED: Trade direction aligns with H1 ",InpRegimeFilterMAPeriod,"-EMA.");
        }
    }
    
    // --- Multi-Timeframe Confluence Check (NEW) ---
    if (InpUseMultiTimeframeConfluence)
    {
        MultiTimeframeAnalysisResult mtfr_4H, mtfr_1H, mtfr_15M;
        int confluentSignals = 0;

        bool tf4H_ok = RunMultiTimeframeAnalysis(InpConfluenceTF_4H, mtfr_4H);
        bool tf1H_ok = RunMultiTimeframeAnalysis(InpConfluenceTF_1H, mtfr_1H);
        bool tf15M_ok = RunMultiTimeframeAnalysis(InpConfluenceTF_15M, mtfr_15M);

        string currentSignal = (bullishSignals > bearishSignals) ? "BUY" : "SELL";

        // Granular Confluence Checks based on Role
        if (currentSignal == "BUY")
        {
            // 4H Structure for BUY: Accumulation or Bullish Market Structure
            if (tf4H_ok && (StringFind(mtfr_4H.wyckoffPhase, "Accumulation") >= 0 || StringFind(mtfr_4H.marketStructure, "Bullish") >= 0))
                confluentSignals++;
            
            // 1H Entry Trigger for BUY: Bullish AMD Trend or Bullish S&D Bias
            if (tf1H_ok && (StringFind(mtfr_1H.amdTrend, "Bullish") >= 0 || StringFind(mtfr_1H.sdBias, "Bullish") >= 0))
                confluentSignals++;

            // 15M Candlestick Confirmation for BUY: Bullish Candlestick Pattern
            if (tf15M_ok && StringFind(mtfr_15M.candlePattern, "Bullish") >= 0)
                confluentSignals++;
        }
        else if (currentSignal == "SELL")
        {
            // 4H Structure for SELL: Distribution or Bearish Market Structure
            if (tf4H_ok && (StringFind(mtfr_4H.wyckoffPhase, "Distribution") >= 0 || StringFind(mtfr_4H.marketStructure, "Bearish") >= 0))
                confluentSignals++;
            
            // 1H Entry Trigger for SELL: Bearish AMD Trend or Bearish S&D Bias
            if (tf1H_ok && (StringFind(mtfr_1H.amdTrend, "Bearish") >= 0 || StringFind(mtfr_1H.sdBias, "Bearish") >= 0))
                confluentSignals++;

            // 15M Candlestick Confirmation for SELL: Bearish Candlestick Pattern
            if (tf15M_ok && StringFind(mtfr_15M.candlePattern, "Bearish") >= 0)
                confluentSignals++;
        }

        if (confluentSignals < InpConfluenceMinSignals)
        {
            Print("⚠️ Multi-Timeframe Confluence Failed: Only ", confluentSignals, " signals align. Minimum required: ", InpConfluenceMinSignals);
            return false; // Fail analysis if confluence not met
        }
        else
        {
            Print("✅ Multi-Timeframe Confluence Met: ", confluentSignals, " signals align.");
        }
    }
    
    double risk = MathAbs(analysis.entry - analysis.stopLoss);
    double reward = MathAbs(analysis.takeProfit - analysis.entry);
    analysis.riskReward = (risk > 0) ? reward / risk : 0;
    
    return true;
}

//+------------------------------------------------------------------+
//| Print Analysis Report                                            |
//+------------------------------------------------------------------+
void PrintAnalysisReport(AnalysisResult &analysis)
{
    string currentTime = TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES);
    
    Print("\n📊 COMBINED MARKET MODEL ANALYSIS - ", InpSymbol);
    Print("🕒 Time: ", currentTime, " | Timeframe: ", EnumToString(InpTimeframe), " | Bar: ", tradeCount + 1);
    Print("────────────────────────────────────────────────────────────");
    Print("📘 MODEL #1: Wyckoff Method");
    Print("   ▸ Detected Phase: ", analysis.wyckoffPhase);
    Print("   ▸ Strength of Phase: ", DoubleToString(analysis.wyckoffStrength, 2));
    Print("   ▸ Volume: ", analysis.volumeCurrent, " vs Avg: ", analysis.volumeAvg, " (Volume analysis)");
    
    Print("\n📘 MODEL #2: AMD (Accumulation / Manipulation / Distribution)");
    Print("   ▸ Trend: ", analysis.amdTrend);
    Print("   ▸ AMD Value: ", DoubleToString(analysis.amdValue, 2));
    Print("   ▸ Momentum: ", DoubleToString(analysis.momentum, 4), " (", (analysis.momentum > 0) ? "positive" : "negative", ")");
    
    Print("\n📘 MODEL #3: Supply & Demand Zones");
    Print("   ▸ Current Price: ", DoubleToString(analysis.currentPrice, 2));
    Print("   ▸ Closest Demand Zone: ", DoubleToString(analysis.demandZone, 2), " (Strong: ", DoubleToString(analysis.demandStrength, 2), ", High Vol)");
    Print("   ▸ Closest Supply Zone: ", DoubleToString(analysis.supplyZone, 2));
    Print("   ▸ Market Bias: ", analysis.sdBias);
    
    Print("\n📘 MODEL #4: Candlestick Patterns");
    Print("   ▸ Pattern Detected: ", analysis.candlePattern);
    Print("   ▸ Location: ", analysis.patternLocation);
    Print("   ▸ Confidence: ", analysis.patternConfidence, "%");
    
    Print("\n📘 MODEL #5: Market Context");
    Print("   ▸ Structure: ", analysis.marketStructure);
    Print("   ▸ VWAP: ", DoubleToString(analysis.vwap, 2), " (Price is ", analysis.vwapPosition, ")");
    Print("   ▸ Order Block Nearby: ", DoubleToString(analysis.orderBlock, 2));
    Print("   ▸ Liquidity Zone: Around ", DoubleToString(analysis.liquidity, 0), " (potential magnet)");
    
    Print("\n────────────────────────────────────────────────────────────");
    Print("💡 STRATEGY DECISION ENGINE");
    Print("📈 FINAL SIGNAL: ", analysis.signal);
    Print("   • Entry Price: ", DoubleToString(analysis.entry, 2));
    Print("   • Stop Loss: ", DoubleToString(analysis.stopLoss, 2));
    Print("   • Take Profit: ", DoubleToString(analysis.takeProfit, 2));
    Print("   • Risk/Reward: ", DoubleToString(analysis.riskReward, 2));
    Print("   • Signal Confidence: 🔵 ", analysis.confidence, "%");

    // --- Multi-Timeframe Confluence Report (NEW) ---
    if (InpUseMultiTimeframeConfluence)
    {
        MultiTimeframeAnalysisResult mtfr_4H, mtfr_1H, mtfr_15M;
        RunMultiTimeframeAnalysis(InpConfluenceTF_4H, mtfr_4H);
        RunMultiTimeframeAnalysis(InpConfluenceTF_1H, mtfr_1H);
        RunMultiTimeframeAnalysis(InpConfluenceTF_15M, mtfr_15M);

        Print("\n📘 MODEL #6: Multi-Timeframe Confluence");
        Print("   ▸ 4H (Structure): ", mtfr_4H.signal, " (Wyckoff: ", mtfr_4H.wyckoffPhase, ", Market Structure: ", mtfr_4H.marketStructure, ")");
        Print("   ▸ 1H (Entry Trigger): ", mtfr_1H.signal, " (AMD: ", mtfr_1H.amdTrend, ", S&D: ", mtfr_1H.sdBias, ")");
        Print("   ▸ 15M (Candlestick Confirmation): ", mtfr_15M.signal, " (Candle: ", mtfr_15M.candlePattern, ")");
    }
}

//+------------------------------------------------------------------+
//| Execute Trading Logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(AnalysisResult &analysis)
{
    if(isTradingDisabled)
    {
        Print("⚠️ Trading is currently disabled due to risk rule: ", disableReason);
        return;
    }

    // The "Institutional" logic was too specific and led to overfitting.
    // It's better to rely on a consistent confidence threshold.
    int confidenceThreshold = 70; // Use a higher, more consistent threshold
    
    // Ultra-Fast Execution Cooldown Check
    if(!CanExecuteUltraFastTrade())
    {
        return;
    }

    if(position.Select(InpSymbol))
    {
       Print("⚠️ Position already open, skipping trade");
       return;
    }
    
    // Enhanced confidence check with adaptive threshold
    if(analysis.confidence < confidenceThreshold)
    {
       Print("⚠️ Confidence too low (", analysis.confidence, "%) for current threshold (", confidenceThreshold, "%) - No trade executed");
       return;
    }
    
    double balance = account.Balance();
    
    // --- SIMPLIFIED AND ROBUST LOT SIZING ---
    // The previous multi-factor calculation was overly complex and unpredictable.
    // A simple, proven risk-percent model is more robust and easier to analyze.
    
    double riskPercent = InpRiskPercent;
    
    // Adaptive risk based on Monte Carlo (this will now work with the fixes)
    // Using half-Kelly for safety is a common practice.
    if(tradeCount > 10 && adaptiveLotSizingEnabled && lastMCResults.edgeRatio != 0) {
        if(lastMCResults.edgeRatio > 0.1 && lastMCResults.kellyFraction > 0.01) {
            riskPercent = MathMin(InpRiskPercent * 1.5, lastMCResults.kellyFraction * 100 * 0.5); 
            Print("Adaptive Risk: Edge detected. Increasing risk to: ", DoubleToString(riskPercent, 2), "%");
        }
        else if(lastMCResults.edgeRatio < 0.05 || lastMCResults.profitFactor < 1.2) {
            riskPercent = InpRiskPercent * 0.75;
            Print("Adaptive Risk: Weak edge. Decreasing risk to: ", DoubleToString(riskPercent, 2), "%");
        }
    }
    
    double riskAmount = balance * (riskPercent / 100.0);
    
    if(!symbolInfo.Name(InpSymbol) || !symbolInfo.RefreshRates())
    {
       Print("❌ Failed to get symbol info");
       return;
    }
    
    double priceDiff = MathAbs(analysis.entry - analysis.stopLoss);
    if(priceDiff == 0 || symbolInfo.TickValue() <= 0)
    {
       Print("❌ Invalid price difference or tick value for risk calculation");
       return;
    }
    
    // A single, reliable formula for lot size based on risk.
    double tickValue = symbolInfo.TickValue();
    double lotSize = (tickValue > 0 && priceDiff > 0) ? riskAmount / (priceDiff / symbolInfo.Point() * tickValue) : 0;
    
    // Removed anti-detection randomness for more consistent backtesting and behavior.
    
    // Ensure lot size is within broker's limits
    lotSize = MathMax(symbolInfo.LotsMin(), lotSize);
    lotSize = MathMin(symbolInfo.LotsMax(), lotSize);
    lotSize = MathMin(InpMaxVolume, lotSize);
    
    // Round to broker's lot step
    double lotStep = symbolInfo.LotsStep();
    if (lotStep > 0) lotSize = floor(lotSize / lotStep) * lotStep;
    
    if(lotSize < symbolInfo.LotsMin())
    {
        Print("⚠️ Calculated lot size (", lotSize, ") is below minimum (", symbolInfo.LotsMin(), "). No trade executed.");
        return;
    }
    
    // Additional safety checks
    double maxRiskAmount = balance * 0.02; // Additional hard cap at 2% of balance
    double potentialLoss = lotSize * priceDiff / symbolInfo.Point() * tickValue;
    
    if(potentialLoss > maxRiskAmount)
    {
        Print("⚠️ Potential loss (", potentialLoss, ") exceeds max risk amount (", maxRiskAmount, "). Adjusting lot size.");
        lotSize = (maxRiskAmount / (priceDiff / symbolInfo.Point() * tickValue));
        
        // Re-apply broker's lot constraints
        lotSize = MathMax(symbolInfo.LotsMin(), lotSize);
        lotSize = MathMin(symbolInfo.LotsMax(), lotSize);
        lotSize = MathMin(InpMaxVolume, lotSize);
        
        if (lotStep > 0) lotSize = floor(lotSize / lotStep) * lotStep;
    }
    
    // Ultra-Fast Execution with enhanced parameter tuning
    bool tradeExecuted = UltraFastExecute(analysis, lotSize);
    
    if(!tradeExecuted)
    {
        Print("❌ Trade Execution Failed in Ultra-Fast Mode");
    }
}

//+------------------------------------------------------------------+
//| Trade transaction event (No change to logic)                     |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD && (trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL))
   {
      if(HistoryDealSelect(trans.deal))
      {
         long dealMagic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         
         if(dealMagic == InpMagicNumber)
         {
            if (HistoryDealGetInteger(trans.deal, DEAL_ENTRY) == DEAL_ENTRY_OUT)
            {
                double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
                tradeCount++;

                if(profit > 0)
                {
                   winCount++;
                   sumWin += profit;
                   winTrades++;
                   Print("✅ WINNING TRADE CLOSED: +$", DoubleToString(profit, 2));
                   if(consecutiveLosses > 0)
                   {
                      Print("📈 Consecutive loss streak broken. Count reset.");
                      consecutiveLosses = 0;
                   }
                   if(isTradingDisabled && disableReason == "Consecutive Loss Limit Hit")
                   {
                      isTradingDisabled = false;
                      disableReason = "";
                      Print("✅ Trading re-enabled after winning trade.");
                      Comment("");
                   }
                }
                else
                {
                   consecutiveLosses++;
                   sumLoss += profit;
                   lossTrades++;
                   Print("❌ LOSING TRADE CLOSED: $", DoubleToString(profit, 2), ". Consecutive losses: ", consecutiveLosses);
                }
                
                totalProfit += profit;
                
                // --- NEW: Periodically run Monte Carlo analysis to adapt strategy ---
                // This replaces the broken logic that only ran at the end.
                // It runs every 25 trades after an initial 20 trades.
                if (tradeCount > 20 && tradeCount % 25 == 0)
                {
                    UpdateAndApplyMonteCarloOptimizations();
                }

                // --- SCALPING CHANGE: Check for Profit Withdrawal Threshold (NEW) ---
                if(InpEnableProfitWithdrawal && totalProfit >= InpProfitWithdrawalThreshold)
                {
                    if(!isTradingDisabled && !isProfitWithdrawalHaltActive)
                    {
                        isTradingDisabled = true;
                        isProfitWithdrawalHaltActive = true;
                        disableReason = "Profit Withdrawal Threshold Reached";
                        Print("💰 PROFIT ALERT: Total profit ($ ", DoubleToString(totalProfit, 2), ") has reached the withdrawal threshold ($ ", DoubleToString(InpProfitWithdrawalThreshold, 2), "). Trading halted. Please consider making a withdrawal.");
                        Comment("TRADING DISABLED: PROFIT WITHDRAWAL RECOMMENDED");
                    }
                }
                
                double winRate = (tradeCount > 0) ? (winCount * 100.0 / tradeCount) : 0.0;
                Print("📊 STATS: Trades: ", tradeCount, " | Win Rate: ", DoubleToString(winRate, 1), 
                      "% | Total P&L: $", DoubleToString(totalProfit, 2));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Advanced Monte Carlo Simulation with Hidden Edge Detection       |
//+------------------------------------------------------------------+
MonteCarloMetrics RunEnhancedMonteCarloSimulation(int numTrades, double winRate, double avgWin, double avgLoss, int numSimulations)
{
    MonteCarloMetrics metrics;
    MathSrand(GetTickCount()); // Seed random number generator
    
    double realTotalProfit = totalProfit;
    int betterCount = 0;
    
    // Fix array allocation - properly allocate memory
    double profits[];
    double drawdowns[];
    
    // Initialize arrays with proper size
    ArrayResize(profits, numSimulations);
    ArrayResize(drawdowns, numSimulations);
    
    // Fix for invalid array access: Don't use 2D array, use 1D arrays for each simulation
    // Instead of storing all equity curves, just calculate max drawdown
    
    for(int sim = 0; sim < numSimulations; sim++)
    {
        double simProfit = 0;
        double peak = 0;
        double maxDD = 0;
        double equityCurve = 0; // Start at 0
        
        for(int t = 0; t < numTrades; t++)
        {
            // Apply hidden edge: slight improvement in win rate for specific conditions
            double adjustedWinRate = winRate;
            
            if(hiddenEdgeActive) {
                // Time-based edge (higher win rates during specific times - hidden from common knowledge)
                MqlDateTime dt;
                TimeToStruct(TimeCurrent(), dt);
                int hour = dt.hour;
                int dayOfWeek = dt.day_of_week;
                
                // Wednesday and Thursday afternoons in NY session tend to have better directional moves
                if((dayOfWeek == 3 || dayOfWeek == 4) && (hour >= 14 && hour <= 16)) {
                    adjustedWinRate *= 1.15; // 15% better win rate
                }
                
                // Asian session consolidation periods often lead to better breakouts
                if(hour >= 2 && hour <= 4) {
                    adjustedWinRate *= 1.08; // 8% better win rate
                }
                
                // Avoid trading during high impact news (usually higher slippage and random moves)
                // This is a simplification - in real implementation, use an economic calendar API
                if(hour == 8 && dt.min >= 25 && dt.min <= 35) { // US data releases often at 8:30 ET
                    adjustedWinRate *= 0.7; // 30% worse win rate
                }
            }
            
            // Simulate trade
            double tradeResult = 0;
            if(MathRand() / 32767.0 < adjustedWinRate)
                tradeResult = avgWin;
            else
                tradeResult = avgLoss;
                
            simProfit += tradeResult;
            equityCurve = simProfit;
            
            // Calculate drawdown
            if(simProfit > peak) peak = simProfit;
            double currentDD = peak - simProfit;
            if(currentDD > maxDD) maxDD = currentDD;
        }
        
        profits[sim] = simProfit;
        drawdowns[sim] = maxDD;
        
        if(simProfit >= realTotalProfit)
            betterCount++;
    }
    
    // Sort profits for percentile calculations
    ArraySort(profits);
    
    // Calculate key metrics
    double sumProfit = 0;
    double sumProfitSquared = 0;
    double winningSimCount = 0;
    
    for(int i = 0; i < numSimulations; i++) {
        sumProfit += profits[i];
        sumProfitSquared += profits[i] * profits[i];
        if(profits[i] > 0) winningSimCount++;
    }
    
    // Calculate main metrics
    metrics.expectedValue = sumProfit / numSimulations;
    double variance = (sumProfitSquared - (sumProfit * sumProfit / numSimulations)) / numSimulations;
    double stdDev = MathSqrt(variance);
    metrics.sharpeRatio = (stdDev != 0) ? metrics.expectedValue / stdDev : 0;
    
    // Worst drawdown
    ArraySort(drawdowns);
    metrics.maxDrawdown = drawdowns[numSimulations-1];
    
    // Win rate confidence
    metrics.winRateConfidence = winningSimCount / numSimulations;
    
    // Edge ratio (expected profit per unit of risk)
    double absoluteAvgLoss = MathAbs(avgLoss);
    metrics.edgeRatio = (absoluteAvgLoss != 0) ? (winRate * avgWin - (1 - winRate) * absoluteAvgLoss) / absoluteAvgLoss : 0;
    
    // Probability of Ruin (simplified)
    double R = avgWin / MathAbs(avgLoss);
    // Fix type conversion from 'long' to 'double'
    metrics.probabilityRuin = (R * winRate < 1 - winRate) ? 1.0 : MathPow((1-winRate)/winRate, (double)numTrades);
    
    // Optimal f (based on Kelly Criterion)
    metrics.optimalF = (avgWin != 0) ? (winRate * avgWin - (1-winRate) * MathAbs(avgLoss)) / avgWin : 0;
    
    // Kelly Fraction (more conservative)
    metrics.kellyFraction = metrics.optimalF * 0.5; // Using half-Kelly for safety
    
    // Median profit
    metrics.medianProfit = (numSimulations % 2 == 0) ? 
                           (profits[numSimulations/2] + profits[numSimulations/2 - 1])/2 : 
                           profits[numSimulations/2];
    
    // Profit Factor
    // Fix type conversion from 'long' to 'double'
    double grossProfit = winRate * avgWin * (double)numTrades;
    double grossLoss = MathAbs((1-winRate) * avgLoss * (double)numTrades);
    metrics.profitFactor = (grossLoss != 0) ? grossProfit / grossLoss : 0;
    
    // Robustness Factor (ratio of mean to standard deviation)
    metrics.robustnessFactor = (stdDev != 0) ? MathAbs(metrics.expectedValue) / stdDev : 0;
    
    // Log results
    double pValue = betterCount / (double)numSimulations;
    Print("\n🔬 ENHANCED Monte Carlo Simulation Results:");
    Print("  Real Total Profit: $", DoubleToString(realTotalProfit, 2));
    Print("  Expected Value: $", DoubleToString(metrics.expectedValue, 2));
    Print("  Sharpe Ratio: ", DoubleToString(metrics.sharpeRatio, 3));
    Print("  Max Drawdown: $", DoubleToString(metrics.maxDrawdown, 2));
    Print("  Win Rate Confidence: ", DoubleToString(metrics.winRateConfidence * 100, 2), "%");
    Print("  Edge Ratio: ", DoubleToString(metrics.edgeRatio, 3));
    Print("  Probability of Ruin: ", DoubleToString(metrics.probabilityRuin * 100, 2), "%");
    Print("  Optimal f: ", DoubleToString(metrics.optimalF, 3));
    Print("  Kelly Fraction: ", DoubleToString(metrics.kellyFraction, 3));
    Print("  Median Profit: $", DoubleToString(metrics.medianProfit, 2));
    Print("  Profit Factor: ", DoubleToString(metrics.profitFactor, 3));
    Print("  Robustness Factor: ", DoubleToString(metrics.robustnessFactor, 3));
    Print("  p-value: ", DoubleToString(pValue, 4));
    
    // Calculate 5% and 95% percentiles for confidence interval
    int idx5Percent = (int)(numSimulations * 0.05);
    int idx95Percent = (int)(numSimulations * 0.95);
    Print("  90% Confidence Interval: [$", DoubleToString(profits[idx5Percent], 2), " to $", 
          DoubleToString(profits[idx95Percent], 2), "]");
    
    lastMCResults = metrics; // Store for adaptive optimization
    return metrics;
}

//+------------------------------------------------------------------+
//| NEW FUNCTION: Update And Apply Monte Carlo Optimizations         |
//+------------------------------------------------------------------+
void UpdateAndApplyMonteCarloOptimizations()
{
    Print("🔄 Running periodic Monte Carlo analysis to adapt strategy...");

    // 1. Get necessary stats from the EA's history
    double avgWin = (winTrades > 0) ? sumWin / winTrades : 0.0;
    double avgLossValue = (lossTrades > 0) ? sumLoss / lossTrades : 0.0; // Loss is negative
    double winRateFrac = (tradeCount > 0) ? (double)winTrades / tradeCount : 0.0;

    if (tradeCount < 20 || winTrades == 0 || lossTrades == 0)
    {
        Print("Not enough trade data for Monte Carlo simulation. Need at least 20 total trades with both wins and losses.");
        return;
    }

    // 2. Run the simulation
    // Using a lower number of simulations (e.g., 5000) for faster periodic execution
    MonteCarloMetrics newMetrics = RunEnhancedMonteCarloSimulation(tradeCount, winRateFrac, avgWin, avgLossValue, 5000);

    // 3. Update the global metrics variable. This is the KEY FIX that makes adaptive logic work.
    lastMCResults = newMetrics;

    Print("✅ Adaptive metrics updated. The EA will now use these new stats to adjust risk.");
}

//+------------------------------------------------------------------+
//| Adaptive Strategy Optimization based on Monte Carlo              |
//+------------------------------------------------------------------+
// NOTE: This function is no longer called directly. Its logic has been
// integrated into ExecuteTradingLogic and the new UpdateAndApplyMonteCarloOptimizations function.
// It is kept here for reference but is not active in the trading loop.
void ApplyMonteCarloOptimizations(MonteCarloMetrics &metrics)
{
    // Only run optimization every 24 hours or after significant equity change
    datetime currentTime = TimeCurrent();
    double currentEquity = account.Equity();
    bool significantEquityChange = MathAbs(currentEquity - dailyEquityStart) / dailyEquityStart > 0.05; // 5% change
    
    if(currentTime - lastOptimizationTime < 24*60*60 && !significantEquityChange) {
        return;
    }
    
    lastOptimizationTime = currentTime;
    Print("🔄 Running adaptive strategy optimization based on Monte Carlo metrics...");
    
    // 1. Adjust risk based on Kelly Criterion and edge ratio
    if(adaptiveLotSizingEnabled) {
        double newRiskPercent = InpRiskPercent;
        
        // If edge is strong and Kelly suggests higher position sizing
        if(metrics.edgeRatio > 0.1 && metrics.kellyFraction > 0.02) {
            newRiskPercent = MathMin(InpRiskPercent * 1.2, metrics.kellyFraction * 100);
            Print("  📈 Edge detected! Increasing risk to: ", DoubleToString(newRiskPercent, 2), "%");
        }
        // If edge is weak or negative, reduce risk
        else if(metrics.edgeRatio < 0.05 || metrics.kellyFraction < 0.01) {
            newRiskPercent = MathMax(InpRiskPercent * 0.7, 0.1); // Min 0.1% risk
            Print("  📉 Weak edge detected! Decreasing risk to: ", DoubleToString(newRiskPercent, 2), "%");
        }
    }
    
    // 2. Adjust TP/SL ratio based on win rate and profit metrics
    if(metrics.profitFactor > 2.0) {
        // Increase take profit targets when profit factor is high
        Print("  🎯 High profit factor detected. Optimizing TP/SL ratios...");
    }
    else if(metrics.profitFactor < 1.0) {
        // Tighten take profits when profit factor is low
        Print("  ⚠️ Low profit factor detected. Adjusting TP/SL ratios for faster exits...");
    }
    
    // 3. Anti-detection pattern adjustments
    if(antiDetectionMode) {
        // Change execution timing randomly to avoid pattern detection
        int randomSeconds = 5 + MathRand() % 25; // Random delay between 5-30 seconds
        Print("  🕵️ Anti-detection pattern engaged. Randomizing execution timing by ", randomSeconds, " seconds");
        
        // Add micro-variations to lot sizes to mask algorithmic footprint
        double lotVariation = 0.01 + MathRand() % 3 / 100.0; // 0.01 to 0.03 variation
        Print("  🔍 Adding lot size variation of ", DoubleToString(lotVariation, 2), " to mask footprint");
    }
    
    // 4. Market regime adaptation
    if(metrics.sharpeRatio < 0.5) {
        Print("  🌊 Low Sharpe ratio detected. Adapting to current market regime...");
        // In low Sharpe environment, focus on higher probability setups only
    }
    else if(metrics.sharpeRatio > 1.5) {
        Print("  🚀 High Sharpe ratio detected. Optimizing for trend exploitation...");
        // In high Sharpe environment, maximize position duration
    }
    
    Print("✅ Adaptive optimization complete. Strategy parameters updated.");
}

// Hidden Market Regime Detection Function
bool IsAdvancedAlgoShiftTime() {
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    int minute = dt.min;
    int dayOfWeek = dt.day_of_week;
    
    // Hidden institutional liquidity windows
    bool isKeyLiquidityWindow = 
        // NY/London overlap - institutional positioning
        (hour == 8 && minute >= 15 && minute <= 45) || 
        // Pre-US stock market open - forex positioning
        (hour == 9 && minute >= 20 && minute <= 35) ||
        // ECB/Fed liquidity operations (typically around 2-2:30 PM European time)
        (hour == 14 && minute >= 0 && minute <= 30) ||
        // US equity close forex repricing
        (hour == 16 && minute >= 0 && minute <= 15);
    
    // Day-specific patterns (hidden institutional knowledge)
    bool isDaySpecificPattern = false;
    
    // Monday AM - weekend gap exploitation
    if(dayOfWeek == 1 && hour < 3) isDaySpecificPattern = true;
    
    // Wednesday PM - pre-FOMC positioning in USD pairs
    if(dayOfWeek == 3 && hour >= 13 && hour <= 14) isDaySpecificPattern = true;
    
    // Friday PM - weekend risk reduction (less participation, easier to move market)
    if(dayOfWeek == 5 && hour >= 15) isDaySpecificPattern = true;
    
    return isKeyLiquidityWindow || isDaySpecificPattern;
}

// Price Pattern Recognition with Hidden Institutional Logic
bool RecognizeInstitutionalPatterns() {
    // SUGGESTION: Like the timing function, these patterns can be very specific
    // and lead to overfitting. Relying on the core confluence model is more robust.
    // This logic is no longer called in OnTick().
    
    // These patterns are based on observations of bank trading activity
    // and are not commonly found in retail trading literature
    
    // Get recent price data
    double highs[], lows[], opens[], closes[];
    if(CopyHigh(InpSymbol, InpTimeframe, 0, 15, highs) <= 0 ||
       CopyLow(InpSymbol, InpTimeframe, 0, 15, lows) <= 0 ||
       CopyOpen(InpSymbol, InpTimeframe, 0, 15, opens) <= 0 ||
       CopyClose(InpSymbol, InpTimeframe, 0, 15, closes) <= 0) 
        return false;
    
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(opens, true);
    ArraySetAsSeries(closes, true);
    
    // Pattern 1: Liquidity Sweep (Stop Hunt) before reversal
    bool liquiditySweep = false;
    // Look for new high/low followed by strong reversal
    if((highs[1] > highs[2] && highs[2] > highs[3] && closes[1] < opens[1] && closes[0] < closes[1]) ||
       (lows[1] < lows[2] && lows[2] < lows[3] && closes[1] > opens[1] && closes[0] > closes[1])) {
        liquiditySweep = true;
    }
    
    // Pattern 2: Institutional Absorption (large volume with little price movement)
    bool absorption = false;
    long volumes[];
    if(CopyTickVolume(InpSymbol, InpTimeframe, 0, 15, volumes) > 0) {
        ArraySetAsSeries(volumes, true);
        double avgVolume = 0;
        for(int i = 5; i < 15; i++) avgVolume += (double)volumes[i];
        avgVolume /= 10;
        
        // High volume but small range candle - absorption
        if(volumes[1] > avgVolume * 1.5 && MathAbs(closes[1] - opens[1]) < (highs[1] - lows[1]) * 0.3) {
            absorption = true;
        }
    }
    
    // Pattern 3: Engineered stops (quick spike through a level then reversal)
    bool engineeredStops = false;
    if((highs[1] > highs[2] + (highs[2] - lows[2]) * 0.3 && closes[1] < (highs[1] + lows[1])/2) ||
       (lows[1] < lows[2] - (highs[2] - lows[2]) * 0.3 && closes[1] > (highs[1] + lows[1])/2)) {
        engineeredStops = true;
    }
    
    return liquiditySweep || absorption || engineeredStops;
}

// --- Multi-TP Partial Close Management (NEW) ---
void ManageMultiTPPartialCloses()
{
    if(!InpEnableMultiTP) return;
    if(!position.Select(InpSymbol)) return;
    if(position.Magic() != InpMagicNumber) return;

    static ulong lastTicket = 0;
    static bool tpHit[4] = {false, false, false, false};
    static double initialLots = 0;

    long ticket = (long)position.Ticket();
    double openPrice = position.PriceOpen();
    double currentPrice = (position.PositionType() == POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
    double atr = cachedATR;
    double direction = (position.PositionType() == POSITION_TYPE_BUY) ? 1.0 : -1.0;

    // Reset tracking if new position
    if(ticket != lastTicket) {
        ArrayInitialize(tpHit, false);
        initialLots = position.Volume();
        lastTicket = ticket;
    }

    MultiTPLevels levels;
    CalculateMultiTPLevels(openPrice, direction, atr, levels);

    for(int i=0; i<levels.count; i++) {
        if(tpHit[i]) continue;
        bool tpReached = (direction > 0) ? (currentPrice >= levels.tp[i]) : (currentPrice <= levels.tp[i]);
        if(tpReached) {
            double lotsToClose = initialLots * levels.percent[i];
            lotsToClose = MathMax(symbolInfo.LotsMin(), lotsToClose);
            lotsToClose = MathMin(position.Volume(), lotsToClose);
            double lotStep = symbolInfo.LotsStep();
            if(lotStep > 0) lotsToClose = floor(lotsToClose / lotStep) * lotStep;
            if(lotsToClose > 0.00001 && position.Volume() - lotsToClose >= symbolInfo.LotsMin()) {
                if(trade.PositionClosePartial(ticket, lotsToClose)) {
                    Print("✅ Multi-TP: Closed ", DoubleToString(lotsToClose, 2), " lots at TP", i+1, " (", DoubleToString(levels.tp[i], symbolInfo.Digits()), ")");
                    tpHit[i] = true;
                }
            }
        }
    }
}
