//+------------------------------------------------------------------+
//|                                        BOT_Scalping_Enhanced.mq5 |
//|                                           Copyright Advanced Trading |
//|                                              https://www.mql5.com|
//+------------------------------------------------------------------+
#property copyright "Advanced Trading"
#property link      "https://www.mql5.com"
#property version   "3.04" // FIX: Version updated after debugging

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\OrderInfo.mqh>

//--- Input parameters
input string  sSymbolSettings = "--- Symbol & Timeframe ---";
input string  InpSymbol = "XAUUSD";               // Trading Symbol
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M1;   // Timeframe for Scalping
input int     InpMagicNumber = 12347;             // Magic Number
input int     InpMinBars = 100;                   // Minimum Bars for Analysis

//--- Risk & Lot Sizing Settings
input string sRiskSettings = "--- Risk & Trade Management ---";
enum ENUM_LOT_SIZING_METHOD
{
    Balance_Proportional,    // Lot size based on a ratio to account balance
    Risk_Percent_Of_Equity,  // Lot size based on Stop Loss and risk %
    Fixed_Lot                // A single, fixed lot size for all trades
};
input ENUM_LOT_SIZING_METHOD InpLotSizingMethod = Balance_Proportional; // Lot Sizing Method
input double InpFixedLot = 0.01;                  // Fixed lot size for the 'Fixed_Lot' method
input double InpLotPer1000Balance = 0.1;          // Lot size per $1000 balance for the 'Balance_Proportional' method
input double InpRiskPercent = 0.5;                // Risk Percent per Trade for the 'Risk_Percent_Of_Equity' method

//--- General Risk
input double InpMaxDailyDrawdownPercent = 5.0;    // Max Daily Drawdown % (0=disable)
input int    InpMaxConsecutiveLosses = 7;         // Max consecutive losses before pause
input bool   InpUseBreakEven = true;              // Use Break-Even?
input int    InpBreakEvenPips = 40;               // Pips in profit to trigger BE
input int    InpBreakEvenLockPips = 5;            // Pips to lock in at BE
input bool   InpUseTrailingStop = true;           // Use Trailing Stop?
input int    InpTrailingStopPips = 60;            // Trailing Stop distance in pips


//--- Advanced Confirmation Filters
input string sAdvancedFilters = "--- Advanced Confirmation Filters ---";
input bool InpUseVWAPFilter = true;           // 1. Use VWAP Filter?
input bool InpUseAVWAPFilter = true;          // 2. Use Anchored VWAP as S/R?
input bool InpUseOBIFilter = false;           // 3. Use Order Book Imbalance? (Requires Broker DOM Data)
input bool InpUseCDFTPFilter = true;          // 4. Use Cumulative Delta & Tick Pressure?
input bool InpUseKCFilter = true;             // 5. Use Keltner Channel Filter?
input bool InpUseSMCFilter = true;            // 6. Use Smart Money Concepts Filter?
input bool InpUseAuctionFilter = false;       // 7. Use Unfinished Auction Filter? (Magnet Zones)
input bool InpUseVolumeSpikeFilter = true;    // 8. Use Tick Volume Spike Detector?

//--- Global variables
CTrade        trade;
CPositionInfo position;
CAccountInfo  account;
CSymbolInfo   symbolInfo;

datetime      lastBarTime = 0;
int           tradeCount = 0;
int           winCount = 0;
double        totalProfit = 0.0;

// --- Risk Management Global Variables
double        dailyEquityStart;
datetime      dayStartDate;
int           consecutiveLosses = 0;
bool          isTradingDisabled = false;
string        disableReason = "";

//--- Analysis structure
struct AnalysisResult
{
    string signal;
    double entry;
    double stopLoss;
    double takeProfit;
    double currentPrice;
    int    confidence;
};

//--- Advanced Analysis structure
struct AdvancedAnalysisResult
{
    double vwap_1m;
    bool   is_above_vwap;
    double avwap_session_high;
    double avwap_session_low;
    string avwap_message;
    double obi_ratio;
    string obi_bias;
    string obi_message;
    double cumulative_delta;
    string cd_divergence;
    long   tick_volume_spike;
    string tick_volume_message;
    double kc_upper;
    double kc_lower;
    string kc_location;
    string smc_zone_message;
    double unfinished_auction_level;
    string auction_message;
    bool   is_trade_confirmed;
    string confirmation_summary;
};

// --- Session tracking globals
datetime session_start_time = 0;
double   session_high_price = 0;
double   session_low_price = 0;
int      session_high_bar_index = 0;
int      session_low_bar_index = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(30);
    
    if(!symbolInfo.Name(InpSymbol))
    {
        Print("❌ Failed to initialize symbol: ", InpSymbol);
        return INIT_FAILED;
    }
    
    symbolInfo.RefreshRates(); // Important to get latest symbol data
    
    dailyEquityStart = account.Equity();
    dayStartDate = TimeCurrent();
    Comment("");

    Print("🚀 MT5 Scalping EA Initialized (v3.04 - Debugged)");
    Print("📊 Symbol: ", InpSymbol, " | Timeframe: ", EnumToString(InpTimeframe));
    Print("💰 Lot Sizing Method: ", EnumToString(InpLotSizingMethod));
    
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
        
        MqlDateTime dt_current, dt_session_start;
        TimeToStruct(TimeCurrent(), dt_current);
        TimeToStruct(session_start_time, dt_session_start);

        if(session_start_time == 0 || dt_current.day_of_year != dt_session_start.day_of_year || dt_current.year != dt_session_start.year) {
            session_start_time = iTime(InpSymbol, PERIOD_D1, 0);
            session_high_price = 0;
            session_low_price = 999999;
        }
        
        if(!isTradingDisabled)
        {
            AnalyzeAndTrade();
        }
    }
}

//+------------------------------------------------------------------+
//| Check main risk management rules                                 |
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
         Print("🚫 RISK ALERT: ", InpMaxConsecutiveLosses, " consecutive losses. All new trading is disabled.");
         Comment("TRADING DISABLED: CONSECUTIVE LOSS LIMIT HIT");
      }
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stops and break-even for open positions          |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
    if(!InpUseBreakEven && !InpUseTrailingStop) return;

    if(position.Select(InpSymbol))
    {
        if(position.Magic() != InpMagicNumber) return;

        double openPrice = position.PriceOpen();
        double currentPrice = (position.PositionType() == POSITION_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();
        double currentSL = position.StopLoss();
        double currentTP = position.TakeProfit();
        long ticket = position.Ticket();
        
        double point = symbolInfo.Point();
        int digits = (int)symbolInfo.Digits();
        
        if(point == 0) return;
        
        double pipsInProfit = 0;

        if (position.PositionType() == POSITION_TYPE_BUY)
        {
            pipsInProfit = (currentPrice - openPrice) / point;
        }
        else
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
                 else
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
//| Main analysis and trading function                               |
//+------------------------------------------------------------------+
void AnalyzeAndTrade()
{
    long bars = SeriesInfoInteger(InpSymbol, InpTimeframe, SERIES_BARS_COUNT);
    if(bars < InpMinBars)
    {
        Print("⚠️ Not enough bars for analysis: ", bars, " < ", InpMinBars);
        return;
    }
    
    AnalysisResult analysis;
    
    int ma_handle = iMA(InpSymbol, InpTimeframe, 14, 0, MODE_SMA, PRICE_CLOSE);
    double ma_buffer[];
    CopyBuffer(ma_handle, 0, 1, 1, ma_buffer);
    IndicatorRelease(ma_handle);
    
    analysis.signal = (iClose(InpSymbol, InpTimeframe, 1) > ma_buffer[0]) ? "✅ BUY" : "❌ SELL";
    analysis.confidence = 75;
    analysis.currentPrice = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);

    int atr_handle = iATR(InpSymbol, InpTimeframe, 14);
    double atr_buffer[];
    CopyBuffer(atr_handle, 0, 1, 1, atr_buffer);
    double atr = atr_buffer[0];
    IndicatorRelease(atr_handle);
    analysis.stopLoss = (StringFind(analysis.signal, "BUY") >= 0) ? analysis.currentPrice - atr * 1.5 : analysis.currentPrice + atr * 1.5;
    analysis.takeProfit = (StringFind(analysis.signal, "BUY") >= 0) ? analysis.currentPrice + atr * 2.5 : analysis.currentPrice - atr * 2.5;

    AdvancedAnalysisResult advanced_analysis;
    RunAdvancedAnalysis(advanced_analysis, analysis.currentPrice, analysis.signal);
    PrintAdvancedAnalysisReport(advanced_analysis, analysis.signal);
    
    ExecuteTradingLogic(analysis, advanced_analysis);
}

//+------------------------------------------------------------------+
//| Execute Trading Logic with Dynamic Lot Sizing                    |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(AnalysisResult &analysis, AdvancedAnalysisResult &advanced_analysis)
{
    if(isTradingDisabled)
    {
        Print("⚠️ Trading is currently disabled due to risk rule: ", disableReason);
        return;
    }

    if(position.Select(InpSymbol))
    {
        Print("⚠️ Position already open for ", InpSymbol, ", skipping new trade signal.");
        return;
    }
    
    if(analysis.confidence < 65)
    {
        Print("⚠️ Core confidence too low (", analysis.confidence, "%) - No trade executed");
        return;
    }
    
    if (!advanced_analysis.is_trade_confirmed)
    {
        Print("❌ TRADE REJECTED by Advanced Filters. Reason: ", advanced_analysis.confirmation_summary);
        return;
    }
    
    Print("✅ TRADE CONFIRMED by Advanced Filters. Proceeding to size and place order...");

    double lotSize = CalculateLotSize(analysis);
    
    if (lotSize <= 0)
    {
        Print("❌ Calculated lot size is zero or invalid. Aborting trade.");
        return;
    }
    
    bool result = false;
    string comment = "AI_AdvScalp_" + EnumToString(InpLotSizingMethod);
   
    if(StringFind(analysis.signal, "BUY") >= 0)
    {
        result = trade.Buy(lotSize, InpSymbol, analysis.currentPrice, analysis.stopLoss, analysis.takeProfit, comment);
    }
    else
    {
        result = trade.Sell(lotSize, InpSymbol, analysis.currentPrice, analysis.stopLoss, analysis.takeProfit, comment);
    }
    
    if(!result)
    {
        Print("❌ Order failed to execute: ", trade.ResultRetcode(), " - ", trade.ResultComment());
    }
    else
    {
        Print("✅ TRADE PLACED: ", trade.ResultOrder(), " | Lot Size: ", lotSize, " | Signal: ", analysis.signal);
    }
}

//+------------------------------------------------------------------+
//| CORRECTED: Calculate Lot Size Based on Selected Method           |
//+------------------------------------------------------------------+
double CalculateLotSize(AnalysisResult &analysis)
{
    double calculated_lot = 0.0;
    symbolInfo.RefreshRates(); // Ensure symbol info is current

    switch(InpLotSizingMethod)
    {
        case Balance_Proportional:
        { // FIX: Added scope for variable declaration
            double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
            if(account_balance <= 0) {
                 Print("❌ Invalid account balance for lot calculation: ", account_balance);
                 return 0.0;
            }
            calculated_lot = (account_balance / 1000.0) * InpLotPer1000Balance;
            break;
        }

        case Risk_Percent_Of_Equity:
        { // FIX: Added scope for variable declaration
            double equity = AccountInfoDouble(ACCOUNT_EQUITY);
            double sl_distance = MathAbs(analysis.currentPrice - analysis.stopLoss);

            if(sl_distance <= 0) {
                Print("❌ Invalid Stop Loss distance (", sl_distance, ") for lot calculation.");
                return 0.0;
            }
            
            double money_to_risk = equity * (InpRiskPercent / 100.0);
            
            double lot_value_at_risk;
            if(!OrderCalcProfit(analysis.signal == "✅ BUY" ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, InpSymbol, 1.0, analysis.currentPrice, analysis.stopLoss, lot_value_at_risk))
            {
                Print("❌ OrderCalcProfit failed with error: ", GetLastError());
                return 0.0;
            }
            
            if(lot_value_at_risk == 0)
            {
                Print("❌ Potential division by zero. Value at risk is zero.");
                return 0.0;
            }
            
            calculated_lot = money_to_risk / MathAbs(lot_value_at_risk);
            break;
        }
            
        case Fixed_Lot:
        { // FIX: Added scope for variable declaration
            calculated_lot = InpFixedLot;
            break;
        }
    }

    // FIX: Using correct function names for CSymbolInfo
    double min_lot = symbolInfo.LotsMin();
    double max_lot = symbolInfo.LotsMax();
    double lot_step = symbolInfo.LotsStep();

    if(lot_step > 0)
    {
       calculated_lot = floor(calculated_lot / lot_step) * lot_step;
    }

    if(calculated_lot < min_lot)
    {
        calculated_lot = min_lot;
        Print("⚠️ Calculated lot was below minimum. Using Min Lot: ", min_lot);
    }
    
    if(max_lot > 0 && calculated_lot > max_lot)
    {
        calculated_lot = max_lot;
        Print("⚠️ Calculated lot was above maximum. Using Max Lot: ", max_lot);
    }
    
    return NormalizeDouble(calculated_lot, 2);
}


//+==================================================================+
//|                   ADVANCED ANALYSIS FUNCTIONS                    |
//+==================================================================+

//+------------------------------------------------------------------+
//| Main function to run all advanced analyses                       |
//+------------------------------------------------------------------+
void RunAdvancedAnalysis(AdvancedAnalysisResult &adv, double currentPrice, string core_signal)
{
    ZeroMemory(adv);
    adv.is_trade_confirmed = true;
    adv.confirmation_summary = "All filters passed.";

    UpdateSessionHighLow();
    
    if(InpUseVWAPFilter || InpUseAVWAPFilter)
    {
        adv.vwap_1m = CalculateVWAP(InpSymbol, PERIOD_M1, 20);
        adv.is_above_vwap = (currentPrice > adv.vwap_1m);
    }

    if(InpUseAVWAPFilter)
    {
        adv.avwap_session_high = CalculateAVWAP(InpSymbol, PERIOD_M1, session_high_bar_index);
        adv.avwap_session_low = CalculateAVWAP(InpSymbol, PERIOD_M1, session_low_bar_index);
        if (adv.avwap_session_high > 0 && currentPrice > adv.avwap_session_high) adv.avwap_message = "Above session high AVWAP (Support)";
        else if (adv.avwap_session_low > 0 && currentPrice < adv.avwap_session_low) adv.avwap_message = "Below session low AVWAP (Resistance)";
        else adv.avwap_message = "Between AVWAP levels";
    }

    if(InpUseOBIFilter)
    {
        adv.obi_ratio = CalculateOBI(InpSymbol, 10, adv.obi_bias);
        if(adv.obi_ratio == DBL_MAX) adv.obi_message = "OBI data not available from broker.";
        else adv.obi_message = "Bias: " + adv.obi_bias + " (" + DoubleToString(adv.obi_ratio*100,1) + "%)";
    }

    if(InpUseCDFTPFilter || InpUseVolumeSpikeFilter)
    {
        CalculateCumulativeDeltaAndSpikes(InpSymbol, adv.cumulative_delta, adv.cd_divergence, adv.tick_volume_spike, adv.tick_volume_message);
    }

    if(InpUseKCFilter)
    {
        CalculateKeltnerChannel(InpSymbol, PERIOD_M1, 20, 1.5, adv.kc_upper, adv.kc_lower);
        if(currentPrice > adv.kc_upper) adv.kc_location = "Above Upper Band";
        else if(currentPrice < adv.kc_lower) adv.kc_location = "Below Lower Band";
        else adv.kc_location = "Inside Channel";
    }

    if(InpUseSMCFilter || InpUseAuctionFilter)
    {
        DetectSMCAndAuctions(InpSymbol, PERIOD_M5, adv.smc_zone_message, adv.unfinished_auction_level, adv.auction_message);
    }

    ConfirmTradeWithFilters(adv, currentPrice, StringFind(core_signal, "BUY") >= 0);
}

//+------------------------------------------------------------------+
//| Final Confirmation Logic                                         |
//+------------------------------------------------------------------+
void ConfirmTradeWithFilters(AdvancedAnalysisResult &adv, double currentPrice, bool isBuySignal)
{
    string rejectionReason = "";
    
    if(isBuySignal)
    {
        if(InpUseVWAPFilter && !adv.is_above_vwap) rejectionReason += "Price below VWAP. ";
        if(InpUseOBIFilter && adv.obi_ratio < 0.15) rejectionReason += "OBI not bullish. ";
        if(InpUseCDFTPFilter && adv.cumulative_delta < 0) rejectionReason += "Negative Cum. Delta. ";
        if(InpUseCDFTPFilter && StringFind(adv.cd_divergence, "Bearish") >= 0) rejectionReason += "Bearish CD Divergence. ";
        if(InpUseSMCFilter && StringFind(adv.smc_zone_message, "Bearish") >= 0) rejectionReason += "In bearish SMC zone. ";
        if(InpUseVolumeSpikeFilter && adv.tick_volume_spike < 0) rejectionReason += "Sell-side volume spike. ";
    }
    else
    {
        if(InpUseVWAPFilter && adv.is_above_vwap) rejectionReason += "Price above VWAP. ";
        if(InpUseOBIFilter && adv.obi_ratio > -0.15) rejectionReason += "OBI not bearish. ";
        if(InpUseCDFTPFilter && adv.cumulative_delta > 0) rejectionReason += "Positive Cum. Delta. ";
        if(InpUseCDFTPFilter && StringFind(adv.cd_divergence, "Bullish") >= 0) rejectionReason += "Bullish CD Divergence. ";
        if(InpUseSMCFilter && StringFind(adv.smc_zone_message, "Bullish") >= 0) rejectionReason += "In bullish SMC zone. ";
        if(InpUseVolumeSpikeFilter && adv.tick_volume_spike > 0) rejectionReason += "Buy-side volume spike. ";
    }
    
    if(rejectionReason != "")
    {
        adv.is_trade_confirmed = false;
        adv.confirmation_summary = rejectionReason;
    }
}


//+------------------------------------------------------------------+
//| 1 & 2. VWAP & Anchored VWAP Calculations                         |
//+------------------------------------------------------------------+
double CalculateVWAP(string symbol, ENUM_TIMEFRAMES timeframe, int period)
{
    double prices[];
    long   volumes[];
    
    if(CopyClose(symbol, timeframe, 1, period, prices) != period || CopyTickVolume(symbol, timeframe, 1, period, volumes) != period)
        return 0.0;
        
    double sum_pv = 0;
    long sum_v = 0;
    
    for(int i=0; i<period; i++)
    {
        sum_pv += prices[i] * volumes[i];
        sum_v += volumes[i];
    }
    
    return (sum_v > 0) ? sum_pv / sum_v : 0.0;
}

void UpdateSessionHighLow()
{
    double high[], low[];
    
    int bars_today = (int)Bars(InpSymbol, InpTimeframe, iTime(InpSymbol, PERIOD_D1, 0), TimeCurrent());
    if (bars_today <= 0) return;

    if(CopyHigh(InpSymbol, InpTimeframe, 0, bars_today, high) > 0 &&
       CopyLow(InpSymbol, InpTimeframe, 0, bars_today, low) > 0)
    {
        int high_idx = ArrayMaximum(high, 0, WHOLE_ARRAY);
        int low_idx = ArrayMinimum(low, 0, WHOLE_ARRAY);
        
        session_high_price = high[high_idx];
        session_low_price = low[low_idx];
        
        session_high_bar_index = (int)Bars(InpSymbol, InpTimeframe) - 1 - high_idx;
        session_low_bar_index = (int)Bars(InpSymbol, InpTimeframe) - 1 - low_idx;
    }
}

double CalculateAVWAP(string symbol, ENUM_TIMEFRAMES timeframe, int anchor_bar_index)
{
    if(anchor_bar_index <= 0) return 0;
    
    int total_bars = (int)Bars(symbol, timeframe);
    if(anchor_bar_index >= total_bars) return 0;

    int bars_to_copy = total_bars - anchor_bar_index;
    if(bars_to_copy <= 0) return 0;

    double prices[];
    long   volumes[];
    
    if(CopyClose(symbol, timeframe, 0, bars_to_copy, prices) != bars_to_copy ||
       CopyTickVolume(symbol, timeframe, 0, bars_to_copy, volumes) != bars_to_copy)
        return 0.0;

    ArraySetAsSeries(prices, true);
    ArraySetAsSeries(volumes, true);

    double sum_pv = 0;
    long sum_v = 0;
    
    for(int i=0; i < bars_to_copy; i++)
    {
        sum_pv += prices[i] * volumes[i];
        sum_v += volumes[i];
    }
    
    return (sum_v > 0) ? sum_pv / sum_v : 0.0;
}

//+------------------------------------------------------------------+
//| 3. Order Book Imbalance (OBI)                                    |
//+------------------------------------------------------------------+
double CalculateOBI(string symbol, int depth, string &bias)
{
    MqlBookInfo book[];
    if(!MarketBookGet(symbol, book))
    {
        return DBL_MAX;
    }
    
    long total_buy_volume = 0;
    long total_sell_volume = 0;
    
    int buy_levels = 0;
    int sell_levels = 0;
    
    for(int i=ArraySize(book)-1; i>=0; i--)
    {
        if(book[i].type == BOOK_TYPE_BUY && buy_levels < depth)
        {
            total_buy_volume += book[i].volume;
            buy_levels++;
        }
        else if(book[i].type == BOOK_TYPE_SELL && sell_levels < depth)
        {
            total_sell_volume += book[i].volume;
            sell_levels++;
        }
    }

    long total_volume = total_buy_volume + total_sell_volume;
    if(total_volume == 0) return 0.0;
    
    double ratio = (double)(total_buy_volume - total_sell_volume) / total_volume;
    
    if (ratio > 0.15) bias = "Strong Bullish";
    else if (ratio > 0) bias = "Slightly Bullish";
    else if (ratio < -0.15) bias = "Strong Bearish";
    else if (ratio < 0) bias = "Slightly Bearish";
    else bias = "Neutral";
    
    return ratio;
}

//+------------------------------------------------------------------+
//| 4 & 8. Cumulative Delta & Volume Spike                           |
//+------------------------------------------------------------------+
void CalculateCumulativeDeltaAndSpikes(string symbol, double &cd_value, string &divergence, long &spike, string &spike_msg)
{
    MqlTick ticks[];
    datetime bar_start_time = iTime(symbol, InpTimeframe, 1);
    int ticks_copied = CopyTicksRange(symbol, ticks, COPY_TICKS_INFO, (ulong)bar_start_time * 1000, (ulong)TimeCurrent() * 1000);
    
    if(ticks_copied <= 0) return;
    
    long delta = 0;
    long total_volume_last_5s = 0;
    long total_volume_prev_periods = 0;
    ulong current_time_ms = TimeCurrent() * 1000;
    
    for(int i=0; i<ticks_copied; i++)
    {
        if((ticks[i].flags & TICK_FLAG_BUY) != 0)
        {
            delta += (long)ticks[i].volume;
        }
        else if((ticks[i].flags & TICK_FLAG_SELL) != 0)
        {
            delta -= (long)ticks[i].volume;
        }
        
        if (current_time_ms - ticks[i].time_msc < 5000)
        {
            total_volume_last_5s += (long)ticks[i].volume;
        } else if (current_time_ms - ticks[i].time_msc < 25000) {
            total_volume_prev_periods += (long)ticks[i].volume;
        }
    }
    
    cd_value = (double)delta;

    double avg_volume_prev = (double)total_volume_prev_periods / 4.0;
    if (avg_volume_prev > 0 && total_volume_last_5s > avg_volume_prev * 2.5) {
        spike = delta > 0 ? total_volume_last_5s : -total_volume_last_5s;
        spike_msg = (delta > 0 ? "BUY" : "SELL") + " side spike detected!";
    } else {
        spike = 0;
        spike_msg = "No significant spike.";
    }

    double price_change = iClose(symbol, InpTimeframe, 1) - iClose(symbol, InpTimeframe, 5);
    if(price_change > 0 && cd_value < 0) divergence = "Bearish Divergence Warning";
    else if(price_change < 0 && cd_value > 0) divergence = "Bullish Divergence Warning";
    else divergence = "No immediate divergence";
}

//+------------------------------------------------------------------+
//| 5. Keltner Channel                                               |
//+------------------------------------------------------------------+
void CalculateKeltnerChannel(string symbol, ENUM_TIMEFRAMES timeframe, int period, double multiplier, double &upper, double &lower)
{
    int ema_handle = iMA(symbol, timeframe, period, 0, MODE_EMA, PRICE_CLOSE);
    int atr_handle = iATR(symbol, timeframe, period);
    
    if(ema_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE) return;
    
    double ema_buffer[], atr_buffer[];
    
    if(CopyBuffer(ema_handle, 0, 1, 1, ema_buffer) > 0 && CopyBuffer(atr_handle, 0, 1, 1, atr_buffer) > 0)
    {
        double ema_value = ema_buffer[0];
        double atr_value = atr_buffer[0];
        upper = ema_value + (atr_value * multiplier);
        lower = ema_value - (atr_value * multiplier);
    }
    
    IndicatorRelease(ema_handle);
    IndicatorRelease(atr_handle);
}

//+------------------------------------------------------------------+
//| 6 & 7. SMC and Unfinished Auction Detection                      |
//+------------------------------------------------------------------+
void DetectSMCAndAuctions(string symbol, ENUM_TIMEFRAMES timeframe, string &smc_msg, double &auction_level, string &auction_msg)
{
    double highs[], lows[], closes[];
    long   volumes[];
    int bars = 50;
    
    if(CopyHigh(symbol, timeframe, 0, bars, highs) != bars || CopyLow(symbol, timeframe, 0, bars, lows) != bars ||
       CopyClose(symbol, timeframe, 0, bars, closes) != bars || CopyTickVolume(symbol, timeframe, 0, bars, volumes) != bars)
        return;
        
    ArraySetAsSeries(highs, true);
    ArraySetAsSeries(lows, true);
    ArraySetAsSeries(closes, true);
    ArraySetAsSeries(volumes, true);

    smc_msg = "No immediate SMC zone";
    for(int i=2; i<bars-1; i++)
    {
        if(lows[i+1] > lows[i] && highs[i-1] < highs[i] && closes[0] > highs[i])
        {
            smc_msg = "Bullish Breaker Zone below";
            break;
        }
        if(highs[i+1] < highs[i] && lows[i-1] > lows[i] && closes[0] < lows[i])
        {
            smc_msg = "Bearish Breaker Zone above";
            break;
        }
    }
    
    double min_low = lows[ArrayMinimum(lows, 0, bars)];
    double max_high = highs[ArrayMaximum(highs, 0, bars)];
    double price_range = max_high - min_low;
    if(price_range == 0) return;
    int num_slices = 20;
    long volume_profile[];
    ArrayResize(volume_profile, num_slices);
    ArrayInitialize(volume_profile, 0);

    double slice_size = price_range / num_slices;
    
    for(int i=0; i<bars; i++)
    {
        int slice_index = (int)((closes[i] - min_low) / slice_size);
        if(slice_index >= 0 && slice_index < num_slices)
        {
            volume_profile[slice_index] += volumes[i];
        }
    }
    
    long min_volume = LONG_MAX;
    int min_index = -1;
    for(int i=0; i<num_slices; i++)
    {
        if(volume_profile[i] > 0 && volume_profile[i] < min_volume)
        {
            min_volume = volume_profile[i];
            min_index = i;
        }
    }
    
    if(min_index != -1)
    {
        auction_level = min_low + (min_index * slice_size) + (slice_size/2);
        auction_msg = "Potential magnet at " + DoubleToString(auction_level, (int)symbolInfo.Digits());
    } else {
        auction_msg = "No clear unfinished auction.";
    }
}

//+------------------------------------------------------------------+
//| Print Advanced Analysis Report                                   |
//+------------------------------------------------------------------+
void PrintAdvancedAnalysisReport(AdvancedAnalysisResult &adv, string core_signal)
{
    Print("\n🔬 ADVANCED FILTER ANALYSIS");
    Print("────────────────────────────────────────────────────────────");
    PrintFormat("Core Signal: %s", core_signal);
    Print("---");

    if(InpUseVWAPFilter) PrintFormat("1. VWAP (1m): %.5f | Price is %s VWAP", adv.vwap_1m, (adv.is_above_vwap ? "ABOVE" : "BELOW"));
    if(InpUseAVWAPFilter) PrintFormat("2. AVWAP Zones: High @ %.5f | Low @ %.5f | %s", adv.avwap_session_high, adv.avwap_session_low, adv.avwap_message);
    if(InpUseOBIFilter) PrintFormat("3. OBI (L2 Data): %s", adv.obi_message);
    if(InpUseCDFTPFilter) PrintFormat("4. Cumulative Delta: %.0f | %s", adv.cumulative_delta, adv.cd_divergence);
    if(InpUseKCFilter) PrintFormat("5. Keltner Channel: U: %.5f | L: %.5f | %s", adv.kc_upper, adv.kc_lower, adv.kc_location);
    if(InpUseSMCFilter) PrintFormat("6. SMC (5m): %s", adv.smc_zone_message);
    if(InpUseAuctionFilter) PrintFormat("7. Unfinished Auction (5m): %s", adv.auction_message);
    if(InpUseVolumeSpikeFilter) PrintFormat("8. Tick Volume Spike: %s", adv.tick_volume_message);
    
    Print("────────────────────────────────────────────────────────────");
    if(adv.is_trade_confirmed)
    {
        Print("✅ FINAL DECISION: TRADE CONFIRMED by advanced filters.");
    }
    else
    {
        Print("❌ FINAL DECISION: TRADE REJECTED. Reason(s): ", adv.confirmation_summary);
    }
    Print("============================================================");
}
