"""
LLM Service for Trading Analysis
Provides integration with Gemini (and extensible to other LLMs) for:
- Technical analysis based on Kronos predictions
- Trading rule enforcement
- Market sentiment analysis
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# Try to import google.generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")


class LLMService:
    """LLM Service for trading analysis using Gemini"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM service

        Args:
            config: Configuration dict containing LLM settings
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.provider = self.llm_config.get('provider', 'gemini')
        self.model_name = self.llm_config.get('model', 'gemini-2.0-flash')
        self.api_key = self.llm_config.get('api_key') or os.environ.get('GEMINI_API_KEY')
        self.model = None
        self.trading_rules = None

        self._init_llm()
        self._load_trading_rules()

    def _init_llm(self):
        """Initialize the LLM client"""
        if not GEMINI_AVAILABLE:
            logger.error("Gemini SDK not available")
            return

        if not self.api_key:
            logger.warning("No Gemini API key configured. Set GEMINI_API_KEY env var or add to config.yaml")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini LLM initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    def _load_trading_rules(self):
        """Load trading rules from file"""
        rules_path = self.llm_config.get('rules_file', 'trading_rules.md')

        # Check multiple possible paths
        possible_paths = [
            rules_path,
            os.path.join(os.path.dirname(__file__), rules_path),
            os.path.join(os.path.dirname(__file__), '..', rules_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.trading_rules = f.read()
                    logger.info(f"Loaded trading rules from: {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load trading rules from {path}: {e}")

        logger.warning("No trading rules file found. Rule-based analysis will be limited.")
        self.trading_rules = "No trading rules configured."

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.model is not None

    async def analyze_technical(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        historical_summary: Dict[str, Any]
    ) -> str:
        """
        Generate technical analysis based on Kronos predictions

        Args:
            symbol: Trading symbol
            prediction_data: Kronos prediction results
            historical_summary: Summary of historical data and indicators

        Returns:
            LLM-generated technical analysis
        """
        if not self.is_available():
            return "LLM service not available. Please configure Gemini API key."

        # Build context for LLM
        current_price = prediction_data.get('current_close', 0)
        p_up = prediction_data.get('p_up_30m', 0)
        exp_return = prediction_data.get('exp_ret_30m', 0)

        # Intraday Technical indicators
        sma_5 = prediction_data.get('sma_5')
        sma_21 = prediction_data.get('sma_21')
        sma_233 = prediction_data.get('sma_233')
        vwap = prediction_data.get('current_vwap')
        bb = prediction_data.get('bollinger_bands', {})

        # Daily Context (fundamental indicators)
        daily_context = prediction_data.get('daily_context', {})
        daily_sma_5 = daily_context.get('daily_sma_5') if daily_context else None
        daily_sma_21 = daily_context.get('daily_sma_21') if daily_context else None
        daily_sma_233 = daily_context.get('daily_sma_233') if daily_context else None
        daily_rsi = daily_context.get('daily_rsi') if daily_context else None
        daily_cci = daily_context.get('daily_cci') if daily_context else None
        daily_trend = daily_context.get('daily_trend', 'Unknown') if daily_context else 'Unknown'
        rsi_signal = daily_context.get('rsi_signal', 'N/A') if daily_context else 'N/A'
        cci_signal = daily_context.get('cci_signal', 'N/A') if daily_context else 'N/A'

        # Percentile predictions
        percentiles = prediction_data.get('percentiles', {})

        # Build daily fundamentals section
        daily_section = ""
        if daily_context:
            daily_section = f"""
## Daily Fundamentals (Higher Timeframe Context)
- **Daily SMA 5 (Bull/Bear Line)**: {f'${daily_sma_5:.2f}' if daily_sma_5 else 'N/A'} {'(Price Above)' if daily_context.get('above_daily_sma5') else '(Price Below)' if daily_context.get('above_daily_sma5') is False else ''}
- **Daily SMA 21**: {f'${daily_sma_21:.2f}' if daily_sma_21 else 'N/A'} {'(Price Above)' if daily_context.get('above_daily_sma21') else '(Price Below)' if daily_context.get('above_daily_sma21') is False else ''}
- **Daily SMA 233**: {f'${daily_sma_233:.2f}' if daily_sma_233 else 'N/A'} {'(Price Above)' if daily_context.get('above_daily_sma233') else '(Price Below)' if daily_context.get('above_daily_sma233') is False else ''}
- **Daily RSI (14)**: {f'{daily_rsi:.1f}' if daily_rsi else 'N/A'} - {rsi_signal}
- **Daily CCI (20)**: {f'{daily_cci:.1f}' if daily_cci else 'N/A'} - {cci_signal}
- **Daily Trend**: {daily_trend}
"""

        prompt = f"""You are an expert quantitative trading analyst. Analyze the following market data and provide actionable insights.

## Current Market Data for {symbol}
- **Current Price**: ${current_price:.2f}
- **Timeframe**: {prediction_data.get('timeframe_minutes', 1)} minutes

## Kronos AI Prediction (Next 30 periods)
- **Probability of Price Increase**: {p_up:.1%}
- **Expected Return**: {exp_return:.2%}
- **Prediction Confidence Bands**:
  - P10 (Bearish): ${percentiles.get('p10', [current_price])[-1]:.2f}
  - P25: ${percentiles.get('p25', [current_price])[-1]:.2f}
  - P50 (Median): ${percentiles.get('p50', [current_price])[-1]:.2f}
  - P75: ${percentiles.get('p75', [current_price])[-1]:.2f}
  - P90 (Bullish): ${percentiles.get('p90', [current_price])[-1]:.2f}
{daily_section}
## Intraday Technical Indicators
- **SMA 5**: {f'${sma_5:.2f}' if sma_5 else 'N/A'} {'(Price Above)' if sma_5 and current_price > sma_5 else '(Price Below)' if sma_5 else ''}
- **SMA 21**: {f'${sma_21:.2f}' if sma_21 else 'N/A'} {'(Price Above)' if sma_21 and current_price > sma_21 else '(Price Below)' if sma_21 else ''}
- **SMA 233**: {f'${sma_233:.2f}' if sma_233 else 'N/A'} {'(Price Above)' if sma_233 and current_price > sma_233 else '(Price Below)' if sma_233 else ''}
- **VWAP**: {f'${vwap:.2f}' if vwap else 'N/A'} {'(Price Above)' if vwap and current_price > vwap else '(Price Below)' if vwap else ''}
- **Bollinger Bands**:
  - Upper: {f'${bb.get("upper", 0):.2f}' if bb else 'N/A'}
  - Middle: {f'${bb.get("middle", 0):.2f}' if bb else 'N/A'}
  - Lower: {f'${bb.get("lower", 0):.2f}' if bb else 'N/A'}

## Analysis Request
Please provide:
1. **Daily Context Assessment**: What does the daily trend, RSI, and CCI tell us about the bigger picture?
2. **Market Condition**: Is the market bullish, bearish, or neutral (considering both daily and intraday)?
3. **Key Levels**: Important support and resistance (both daily and intraday levels)
4. **Signal Strength**: How strong is the Kronos prediction signal? (Weak/Moderate/Strong)
5. **Risk Assessment**: What are the key risks to watch? (RSI overbought/oversold, CCI extremes)
6. **Recommendation**: Clear actionable suggestion (Buy/Sell/Hold/Wait)

Keep the analysis concise but insightful. Use bullet points for clarity."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating technical analysis: {e}")
            return f"Error generating analysis: {str(e)}"

    async def check_trading_rules(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        proposed_action: Optional[str] = None
    ) -> str:
        """
        Check if proposed action or current situation aligns with trading rules

        Args:
            symbol: Trading symbol
            prediction_data: Current prediction data
            proposed_action: Optional proposed trade (e.g., "BUY", "SELL")

        Returns:
            LLM analysis of rule compliance
        """
        if not self.is_available():
            return "LLM service not available. Please configure Gemini API key."

        current_price = prediction_data.get('current_close', 0)
        p_up = prediction_data.get('p_up_30m', 0)

        # Get current time info
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        day_of_week = now.strftime('%A')
        time_str = now.strftime('%H:%M')

        prompt = f"""You are a trading discipline coach. Your job is to help the trader follow their rules strictly.

## Trading Rules
{self.trading_rules}

## Current Situation
- **Symbol**: {symbol}
- **Current Price**: ${current_price:.2f}
- **Day**: {day_of_week}
- **Time (ET)**: {time_str}
- **Kronos P(Up)**: {p_up:.1%}
- **Proposed Action**: {proposed_action or 'None - Just checking current position'}

## Your Task
1. Review the trading rules above
2. Assess if the current situation or proposed action follows the rules
3. If rules would be violated, clearly explain which rule and why
4. Provide a PASS/WARN/FAIL status for rule compliance
5. Give specific advice to help the trader stay disciplined

Be direct and firm - the trader needs honest feedback to stay disciplined."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error checking trading rules: {e}")
            return f"Error checking rules: {str(e)}"

    async def analyze_sentiment(
        self,
        symbol: str,
        news_items: List[Dict[str, Any]]
    ) -> str:
        """
        Analyze market sentiment from news

        Args:
            symbol: Trading symbol
            news_items: List of news articles with headline, summary, source

        Returns:
            Sentiment analysis summary
        """
        if not self.is_available():
            return "LLM service not available. Please configure Gemini API key."

        if not news_items:
            return "No recent news available for sentiment analysis."

        # Format news for prompt
        news_text = ""
        for i, news in enumerate(news_items[:10], 1):  # Limit to 10 items
            headline = news.get('headline', 'No headline')
            summary = news.get('summary', '')[:200]  # Truncate long summaries
            source = news.get('source', 'Unknown')
            created = news.get('created_at', '')
            news_text += f"{i}. [{source}] {headline}\n   {summary}...\n   Time: {created}\n\n"

        prompt = f"""You are a market sentiment analyst. Analyze the following news for {symbol} and provide a sentiment assessment.

## Recent News
{news_text}

## Analysis Required
1. **Overall Sentiment**: Bullish / Bearish / Neutral (with confidence level)
2. **Key Themes**: What are the main topics driving sentiment?
3. **Market Impact**: How might this news affect {symbol}'s price?
4. **Sentiment Score**: Rate from -10 (extremely bearish) to +10 (extremely bullish)
5. **Notable Headlines**: Which 1-2 headlines are most market-moving?

Provide a concise but comprehensive sentiment summary."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return f"Error analyzing sentiment: {str(e)}"

    async def generate_daily_highlights(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        news_items: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a concise daily highlights summary for intraday trading

        Args:
            symbol: Trading symbol
            prediction_data: Kronos prediction results with technical indicators
            news_items: Recent news for sentiment

        Returns:
            Concise highlights summary
        """
        if not self.is_available():
            return "LLM service not available. Please configure Gemini API key."

        # Extract key data
        current_price = prediction_data.get('current_close', 0)
        p_up = prediction_data.get('p_up_30m', 0)
        exp_return = prediction_data.get('exp_ret_30m', 0)

        # Intraday Technical indicators (from current timeframe)
        sma_5 = prediction_data.get('sma_5')
        sma_21 = prediction_data.get('sma_21')
        sma_233 = prediction_data.get('sma_233')
        vwap = prediction_data.get('current_vwap')
        bb = prediction_data.get('bollinger_bands', {})

        # Daily Context (fundamental indicators for intraday)
        daily_context = prediction_data.get('daily_context', {})
        daily_sma_5 = daily_context.get('daily_sma_5') if daily_context else None
        daily_sma_21 = daily_context.get('daily_sma_21') if daily_context else None
        daily_sma_233 = daily_context.get('daily_sma_233') if daily_context else None
        daily_rsi = daily_context.get('daily_rsi') if daily_context else None
        daily_cci = daily_context.get('daily_cci') if daily_context else None
        daily_trend = daily_context.get('daily_trend', 'Unknown') if daily_context else 'Unknown'
        rsi_signal = daily_context.get('rsi_signal', 'N/A') if daily_context else 'N/A'
        cci_signal = daily_context.get('cci_signal', 'N/A') if daily_context else 'N/A'

        # Daily price levels
        prev_day_high = daily_context.get('prev_day_high') if daily_context else None
        prev_day_low = daily_context.get('prev_day_low') if daily_context else None
        prev_day_close = daily_context.get('prev_day_close') if daily_context else None
        today_high = daily_context.get('today_high') if daily_context else None
        today_low = daily_context.get('today_low') if daily_context else None
        three_day_high = daily_context.get('three_day_high') if daily_context else None
        three_day_low = daily_context.get('three_day_low') if daily_context else None

        # Percentiles for support/resistance
        percentiles = prediction_data.get('percentiles', {})

        # Format news summary
        news_summary = "No recent news."
        if news_items and len(news_items) > 0:
            headlines = [f"- {n.get('headline', '')[:100]}" for n in news_items[:5]]
            news_summary = "\n".join(headlines)

        # Build daily fundamentals section - Daily SMA5 is #1 MOST IMPORTANT
        daily_fundamentals = f"""## 🔥 #1 MOST IMPORTANT: DAILY SMA 5 (Bull/Bear Line)
- **Daily SMA 5**: {f'${daily_sma_5:.2f}' if daily_sma_5 else 'N/A'} - Price {'ABOVE ✅ BULLISH BIAS' if daily_context and daily_context.get('above_daily_sma5') else 'BELOW ❌ BEARISH BIAS' if daily_context and daily_context.get('above_daily_sma5') is False else 'N/A'}
- THIS IS THE KEY LEVEL FOR INTRADAY DIRECTION. Above = look for longs. Below = look for shorts.

## 🎯 #2 IMPORTANT: INTRADAY VWAP & BOLLINGER BANDS
- **VWAP**: {f'${vwap:.2f}' if vwap else 'N/A'} - Price {'ABOVE (Bullish intraday)' if vwap and current_price > vwap else 'BELOW (Bearish intraday)' if vwap else 'N/A'}
- **Bollinger Upper**: {f'${bb.get("upper", 0):.2f}' if bb else 'N/A'} (Overbought/Resistance)
- **Bollinger Lower**: {f'${bb.get("lower", 0):.2f}' if bb else 'N/A'} (Oversold/Support)
- VWAP acts as intraday magnet. Bollinger bands show volatility extremes.

## 📊 DAILY CONTEXT (Reference)
- **Daily SMA 21**: {f'${daily_sma_21:.2f}' if daily_sma_21 else 'N/A'} - Price {'ABOVE' if daily_context and daily_context.get('above_daily_sma21') else 'BELOW' if daily_context and daily_context.get('above_daily_sma21') is False else 'N/A'}
- **Daily SMA 233**: {f'${daily_sma_233:.2f}' if daily_sma_233 else 'N/A'} - Price {'ABOVE' if daily_context and daily_context.get('above_daily_sma233') else 'BELOW' if daily_context and daily_context.get('above_daily_sma233') is False else 'N/A'}
- **Daily RSI (14)**: {f'{daily_rsi:.1f}' if daily_rsi else 'N/A'} - {rsi_signal}
- **Daily CCI (20)**: {f'{daily_cci:.1f}' if daily_cci else 'N/A'} - {cci_signal}
- **Overall Daily Trend**: {daily_trend}

## 📍 KEY DAILY PRICE LEVELS (Support/Resistance Reference)
- **Previous Day High**: {f'${prev_day_high:.2f}' if prev_day_high else 'N/A'} | **Previous Day Low**: {f'${prev_day_low:.2f}' if prev_day_low else 'N/A'} | **Prev Close**: {f'${prev_day_close:.2f}' if prev_day_close else 'N/A'}
- **3-Day High**: {f'${three_day_high:.2f}' if three_day_high else 'N/A'} | **3-Day Low**: {f'${three_day_low:.2f}' if three_day_low else 'N/A'}
- **Today's Range**: High {f'${today_high:.2f}' if today_high else 'N/A'} / Low {f'${today_low:.2f}' if today_low else 'N/A'}"""

        prompt = f"""You are an expert intraday trading analyst. Based on the data below, provide a CONCISE daily highlights summary.

## Trading Rules Context
{self.trading_rules}

## Current Market Data for {symbol}
- **Current Price**: ${current_price:.2f}
- **Kronos P(Up)**: {p_up:.1%}
- **Expected Return**: {exp_return:.2%}

{daily_fundamentals}

## Intraday Technical Indicators (Current Timeframe)
- **Intraday SMA 5**: {f'${sma_5:.2f}' if sma_5 else 'N/A'} - Price is {'ABOVE' if sma_5 and current_price > sma_5 else 'BELOW' if sma_5 else 'N/A'}
- **Intraday SMA 21**: {f'${sma_21:.2f}' if sma_21 else 'N/A'} - Price is {'ABOVE' if sma_21 and current_price > sma_21 else 'BELOW' if sma_21 else 'N/A'}
- **Intraday SMA 233**: {f'${sma_233:.2f}' if sma_233 else 'N/A'}
- **VWAP**: {f'${vwap:.2f}' if vwap else 'N/A'} - Price is {'ABOVE' if vwap and current_price > vwap else 'BELOW' if vwap else 'N/A'}
- **Bollinger Upper**: {f'${bb.get("upper", 0):.2f}' if bb else 'N/A'}
- **Bollinger Lower**: {f'${bb.get("lower", 0):.2f}' if bb else 'N/A'}

## Kronos Prediction Levels (Next 30 bars)
- **P90 (Bullish Target)**: ${percentiles.get('p90', [current_price])[-1]:.2f}
- **P75**: ${percentiles.get('p75', [current_price])[-1]:.2f}
- **P50 (Median)**: ${percentiles.get('p50', [current_price])[-1]:.2f}
- **P25**: ${percentiles.get('p25', [current_price])[-1]:.2f}
- **P10 (Bearish Target)**: ${percentiles.get('p10', [current_price])[-1]:.2f}

## Recent News Headlines
{news_summary}

---

## YOUR TASK: Provide a BRIEF Daily Highlights summary with these sections:

### 1. 🔥 DAILY SMA 5 STATUS (MOST IMPORTANT - 1-2 lines)
State clearly: Price is ABOVE/BELOW Daily SMA 5
This determines your intraday bias: ABOVE = bullish bias (look for longs), BELOW = bearish bias (look for shorts)

### 2. 🎯 VWAP & BOLLINGER STATUS (1-2 lines)
Where is price vs VWAP? Near upper or lower Bollinger band?
This determines entry timing and mean reversion potential.

### 3. SENTIMENT (1 line)
State: BULLISH / BEARISH / NEUTRAL with confidence level

### 4. DAY TYPE (1-2 lines)
Based on Kronos signal strength:
- If P(up) > 70% or < 30%: Likely TRENDING DAY
- If P(up) 45-55%: Likely RANGE/SWING DAY

### 5. KEY LEVELS (bullet points - prioritized)
- 🔥 **Daily SMA 5**: THE bull/bear line (most important)
- 🎯 **VWAP**: Intraday pivot level
- 🎯 **Bollinger Bands**: Upper (resistance) / Lower (support)
- Previous Day High/Low: Key intraday S/R
- 3-Day High/Low: Swing S/R reference

### 6. TRADE BIAS (1-2 lines)
Based on Daily SMA 5 position + VWAP, what's the recommended bias?
Be specific: "Long bias above Daily SMA5, fade at Bollinger Upper" etc.

Keep it SHORT and ACTIONABLE. Daily SMA 5 is THE key level."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating daily highlights: {e}")
            return f"Error generating highlights: {str(e)}"

    async def get_full_analysis(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        news_items: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Get complete analysis including technical, rules, and sentiment

        Returns:
            Dict with 'technical', 'rules', 'sentiment' keys
        """
        results = {}

        # Technical analysis
        results['technical'] = await self.analyze_technical(
            symbol, prediction_data, {}
        )

        # Trading rules check
        results['rules'] = await self.check_trading_rules(
            symbol, prediction_data
        )

        # Sentiment analysis (if news provided)
        if news_items:
            results['sentiment'] = await self.analyze_sentiment(
                symbol, news_items
            )
        else:
            results['sentiment'] = "No news data provided for sentiment analysis."

        return results
