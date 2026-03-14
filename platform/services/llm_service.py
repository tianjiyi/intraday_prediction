"""
LLM Service for Trading Analysis
Provides integration with Claude (Anthropic) for:
- Technical analysis based on Kronos predictions
- Trading rule enforcement
- Market sentiment analysis
"""

import os
import base64
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)

# Create a separate logger for LLM context logging
llm_context_logger = logging.getLogger('llm_context')
llm_context_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# File handler for LLM context (rotates daily)
llm_log_file = LOG_DIR / 'llm_context.log'
llm_file_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
llm_file_handler.setLevel(logging.DEBUG)
llm_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
llm_context_logger.addHandler(llm_file_handler)

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")

# For Ollama native API
import httpx


class LLMService:
    """LLM Service for trading analysis using Claude (Anthropic)"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM service

        Args:
            config: Configuration dict containing LLM settings
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.provider = self.llm_config.get('provider', 'anthropic')
        self.model_name = self.llm_config.get('model', 'claude-opus-4-6')
        self.base_url = self.llm_config.get('base_url', None)
        self.api_key = os.environ.get('ANTHROPIC_API_KEY') or self.llm_config.get('api_key')
        self.client = None
        self.trading_rules = None
        self.supports_vision = self.provider == 'anthropic'

        self._init_llm()
        self._load_trading_rules()

    def _init_llm(self):
        """Initialize the LLM client based on provider"""
        if self.provider == 'ollama':
            try:
                base_url = self.base_url or 'http://localhost:11434'
                self.client = httpx.Client(base_url=base_url, timeout=120.0)
                # Test connection
                resp = self.client.get('/api/tags')
                resp.raise_for_status()
                logger.info(f"Ollama LLM initialized: {self.model_name} at {base_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                self.client = None

        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic SDK not available")
                return
            if not self.api_key:
                logger.warning("No Anthropic API key configured. Set ANTHROPIC_API_KEY env var.")
                return
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic LLM initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")

        else:
            logger.error(f"Unknown LLM provider: {self.provider}")

    def _generate(self, prompt: str, image_base64: Optional[str] = None, max_tokens: int = 4096) -> str:
        """
        Unified generation method — calls LLM API with text and optional image.

        Args:
            prompt: The text prompt
            image_base64: Optional base64-encoded PNG image (Anthropic only)
            max_tokens: Max tokens in response

        Returns:
            Response text from LLM
        """
        if self.provider == 'ollama':
            if image_base64:
                logger.debug("Image provided but Ollama/Qwen does not support vision — skipping image")
            resp = self.client.post('/api/chat', json={
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'think': False,
                'options': {'num_predict': max_tokens},
            })
            resp.raise_for_status()
            return resp.json()['message']['content']

        # Anthropic provider (supports multimodal)
        content = []

        if image_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64,
                },
            })

        content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text

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
        return self.client is not None

    def _log_llm_context(self, method_name: str, prompt: str, response: str = None, error: str = None):
        """
        Log LLM input/output context to file

        Args:
            method_name: Name of the calling method (e.g., 'analyze_technical')
            prompt: The prompt sent to the LLM
            response: The response received from the LLM (if successful)
            error: Error message (if failed)
        """
        separator = "=" * 80

        # Log input
        llm_context_logger.info(f"\n{separator}")
        llm_context_logger.info(f"METHOD: {method_name}")
        llm_context_logger.info(f"MODEL: {self.model_name}")
        llm_context_logger.info(f"{separator}")
        llm_context_logger.info(f">>> INPUT PROMPT:\n{prompt}")

        # Log output or error
        if error:
            llm_context_logger.error(f"<<< ERROR: {error}")
        elif response:
            llm_context_logger.info(f"<<< OUTPUT RESPONSE:\n{response}")

        llm_context_logger.info(f"{separator}\n")

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
            response_text = self._generate(prompt)
            self._log_llm_context('analyze_technical', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating technical analysis: {e}")
            self._log_llm_context('analyze_technical', prompt, error=error_msg)
            return f"Error generating analysis: {error_msg}"

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
            response_text = self._generate(prompt)
            self._log_llm_context('check_trading_rules', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error checking trading rules: {e}")
            self._log_llm_context('check_trading_rules', prompt, error=error_msg)
            return f"Error checking rules: {error_msg}"

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
            response_text = self._generate(prompt)
            self._log_llm_context('analyze_sentiment', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing sentiment: {e}")
            self._log_llm_context('analyze_sentiment', prompt, error=error_msg)
            return f"Error analyzing sentiment: {error_msg}"

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
            response_text = self._generate(prompt)
            self._log_llm_context('generate_daily_highlights', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating daily highlights: {e}")
            self._log_llm_context('generate_daily_highlights', prompt, error=error_msg)
            return f"Error generating highlights: {error_msg}"

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

    async def chat_with_context(
        self,
        symbol: str,
        user_message: str,
        prediction_data: Dict[str, Any],
        chat_history: List[Dict[str, str]] = None,
        chart_screenshot: Optional[str] = None,
        memory_context: str = "",
        news_context: str = "",
        chart_state: Optional[Dict[str, Any]] = None,
        market_env_context: str = "",
        trade_ctx_context: str = "",
    ) -> str:
        """
        Chat with the LLM using current market context

        Args:
            symbol: Trading symbol
            user_message: User's chat message
            prediction_data: Current prediction data with indicators
            chat_history: Previous chat messages for context
            chart_screenshot: Base64-encoded PNG screenshot of the chart (optional)
            memory_context: Pre-built agent memory context string (from AgentMemoryService)
            chart_state: Frontend chart indicator state (day trading VWAP etc.)

        Returns:
            LLM response
        """
        if not self.is_available():
            return "LLM service not available. Please configure Gemini API key."

        # Build market context
        current_price = prediction_data.get('current_close', 0)
        p_up = prediction_data.get('p_up_30m', 0)
        exp_return = prediction_data.get('exp_ret_30m', 0)

        # Intraday indicators
        sma_5 = prediction_data.get('sma_5')
        sma_21 = prediction_data.get('sma_21')
        sma_233 = prediction_data.get('sma_233')
        vwap = prediction_data.get('current_vwap')
        bb = prediction_data.get('bollinger_bands', {})

        # Daily context
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
        three_day_high = daily_context.get('three_day_high') if daily_context else None
        three_day_low = daily_context.get('three_day_low') if daily_context else None

        # Percentiles
        percentiles = prediction_data.get('percentiles', {})

        # Check if frontend day trading VWAP supersedes backend VWAP
        dt_has_vwap = (
            chart_state
            and chart_state.get("day_trading", {}).get("enabled")
            and chart_state.get("day_trading", {}).get("vwap")
        )

        # Get current time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        day_of_week = now.strftime('%A')
        time_str = now.strftime('%H:%M')

        # Format chat history
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_text = "\n## Recent Conversation:\n"
            for msg in chat_history[-6:]:  # Last 6 messages
                role = "User" if msg.get('role') == 'user' else "Assistant"
                history_text += f"**{role}**: {msg.get('content', '')[:500]}\n\n"

        # Build context
        market_context = f"""## Current Market Data for {symbol}
- **Current Price**: ${current_price:.2f}
- **Day/Time**: {day_of_week} {time_str} ET
- **Timeframe**: {prediction_data.get('timeframe_minutes', 1)} minutes

## Kronos AI Prediction (Next 30 periods)
- **Probability of Price Increase**: {p_up:.1%}
- **Expected Return**: {exp_return:.2%}
- **Prediction Targets**:
  - P90 (Bullish): ${percentiles.get('p90', [current_price])[-1]:.2f}
  - P50 (Median): ${percentiles.get('p50', [current_price])[-1]:.2f}
  - P10 (Bearish): ${percentiles.get('p10', [current_price])[-1]:.2f}

## 🔥 KEY LEVEL - Daily SMA 5 (Bull/Bear Line)
- **Daily SMA 5**: {f'${daily_sma_5:.2f}' if daily_sma_5 else 'N/A'} - Price is {'ABOVE ✅ (Bullish bias)' if daily_context and daily_context.get('above_daily_sma5') else 'BELOW ❌ (Bearish bias)' if daily_context and daily_context.get('above_daily_sma5') is False else 'N/A'}

## Intraday Indicators
{f'- **VWAP**: ${vwap:.2f} {"(Price Above)" if current_price > vwap else "(Price Below)"}' if vwap and not dt_has_vwap else '- **VWAP**: (see Day Trading Indicators below)' if dt_has_vwap else '- **VWAP**: N/A'}
{f'- **Bollinger Upper**: ${bb.get("upper", 0):.2f}' if bb and not dt_has_vwap else '- **Bollinger Bands**: (see Day Trading VWAP bands below)' if dt_has_vwap else '- **Bollinger Upper**: N/A'}
{f'- **Bollinger Lower**: ${bb.get("lower", 0):.2f}' if bb and not dt_has_vwap else ''}
- **Intraday SMA 5**: {f'${sma_5:.2f}' if sma_5 else 'N/A'}
- **Intraday SMA 21**: {f'${sma_21:.2f}' if sma_21 else 'N/A'}
- **Intraday SMA 233**: {f'${sma_233:.2f}' if sma_233 else 'N/A'}

## Daily Fundamentals
- **Daily SMA 21**: {f'${daily_sma_21:.2f}' if daily_sma_21 else 'N/A'}
- **Daily SMA 233**: {f'${daily_sma_233:.2f}' if daily_sma_233 else 'N/A'}
- **Daily RSI (14)**: {f'{daily_rsi:.1f}' if daily_rsi else 'N/A'} - {rsi_signal}
- **Daily CCI (20)**: {f'{daily_cci:.1f}' if daily_cci else 'N/A'} - {cci_signal}
- **Daily Trend**: {daily_trend}

## Key Daily Price Levels
- **Prev Day High**: {f'${prev_day_high:.2f}' if prev_day_high else 'N/A'} | **Prev Day Low**: {f'${prev_day_low:.2f}' if prev_day_low else 'N/A'}
- **3-Day High**: {f'${three_day_high:.2f}' if three_day_high else 'N/A'} | **3-Day Low**: {f'${three_day_low:.2f}' if three_day_low else 'N/A'}
"""

        # Add visual context note if screenshot is available
        visual_context = ""
        if chart_screenshot:
            visual_context = """
## Visual Context
A screenshot of the current chart is attached. Use this visual information to:
- Identify candlestick patterns, trends, and price action
- Reference what you see on the chart when answering
- Point out any visual patterns the trader should notice
"""

        # Inject agent memory context if available
        memory_section = ""
        if memory_context:
            memory_section = f"\n{memory_context}\n"

        # Inject critical news context if available
        news_section = ""
        if news_context:
            news_section = f"\n{news_context}\n"

        # Inject day trading chart state if available
        dt_section = ""
        if chart_state:
            dt = chart_state.get("day_trading", {})
            if dt.get("enabled") and dt.get("vwap"):
                v = dt["vwap"]
                dt_section = f"""
## Day Trading Indicators (Frontend Session VWAP)
- **Session VWAP**: ${v.get('value', 0):.2f}
- **VWAP StdDev**: {v.get('std', 0):.4f}
- **+1σ Band**: ${v.get('upper1', 0):.2f} / **-1σ Band**: ${v.get('lower1', 0):.2f}
- **+2σ Band**: ${v.get('upper2', 0):.2f} / **-2σ Band**: ${v.get('lower2', 0):.2f}
- **Timeframe**: {dt.get('timeframe_minutes', 1)}m
Note: These VWAP values are computed from the frontend chart and reflect the exact indicators the user sees.
"""

        # Inject visible time range for drawing commands
        time_range_section = ""
        if chart_state:
            vtr = chart_state.get("visible_time_range")
            if vtr and vtr.get("from") and vtr.get("to"):
                from datetime import datetime as _dt, timezone as _tz
                try:
                    t_from = int(vtr["from"])
                    t_to = int(vtr["to"])
                    # Provide both unix timestamps and human-readable times
                    dt_from = _dt.fromtimestamp(t_from, tz=_tz.utc).strftime('%Y-%m-%d %H:%M')
                    dt_to = _dt.fromtimestamp(t_to, tz=_tz.utc).strftime('%Y-%m-%d %H:%M')
                    time_range_section = f"""
## Chart Visible Time Range
- From: {t_from} ({dt_from} UTC) | To: {t_to} ({dt_to} UTC)
- Use these Unix timestamps as reference when drawing boxes (zone with startTime/endTime) or trendlines.
- Left ~25% of chart: ~{t_from} | Center: ~{(t_from + t_to) // 2} | Right ~75%: ~{t_from + 3 * (t_to - t_from) // 4}
"""
                except (ValueError, TypeError):
                    pass

        # Inject current chart drawings context
        drawings_section = ""
        if chart_state:
            drawings = chart_state.get("drawings", [])
            if drawings:
                lines = []
                for d in drawings:
                    dtype = d.get("type", "unknown")
                    src = d.get("source", "unknown")
                    lbl = d.get("label", "")
                    if dtype == "hline":
                        lines.append(f"- {src} hline at ${d.get('price', 0):.2f}{f' ({lbl})' if lbl else ''}")
                    elif dtype == "trendline":
                        lines.append(f"- {src} trendline from ${d.get('startPrice', 0):.2f} to ${d.get('endPrice', 0):.2f}{f' ({lbl})' if lbl else ''}")
                    elif dtype == "zone":
                        lines.append(f"- {src} zone ${d.get('priceLow', 0):.2f}-${d.get('priceHigh', 0):.2f}{f' ({lbl})' if lbl else ''}")
                drawings_section = "\n## Current Chart Drawings:\n" + "\n".join(lines) + "\n"

        # Build market environment and trade context sections
        market_env_section = f"\n{market_env_context}\n" if market_env_context else ""
        trade_ctx_section = f"\n{trade_ctx_context}\n" if trade_ctx_context else ""

        prompt = f"""You are an expert intraday trading assistant with access to real-time market data. You help traders make informed decisions based on technical analysis.

{market_context}
{dt_section}
{time_range_section}
{drawings_section}
{memory_section}
{news_section}
{market_env_section}{trade_ctx_section}
{visual_context}
{history_text}
## User's Question:
{user_message}

## Instructions:
1. Answer the user's question based on the current market data
2. Be specific with price levels and percentages
3. Keep responses concise but informative
4. If suggesting trades, always mention key levels (Daily SMA 5, VWAP, Bollinger Bands)
5. Use the Kronos prediction probabilities to assess signal strength
6. Daily SMA 5 is the most important level for intraday bias
7. If a chart screenshot is provided, reference what you see visually to support your analysis
8. When critical market catalysts are present above, structure your analysis as: **Catalyst** > **Transmission Path** > **Affected Sectors** > **Market Impact Horizon** > **Key Levels** > **Invalidation Conditions**
9. If no critical catalysts are present, briefly state "No major market-moving catalyst detected" before answering
10. Do not offer generic market opinions not tied to a scored catalyst or technical level
11. Use the Market Environment context to frame your analysis — if risk_mode is "risk_off", bias toward defensive/cautious recommendations; if "risk_on", be more constructive
12. When Trade Context shows an imminent event (e.g., "CPI in 42m"), warn the trader about positioning risk and suggest reducing size or waiting
13. Reference active themes when relevant to the symbol being discussed

## Drawing Capabilities:
You can draw on the chart! When the user asks you to mark price levels, support, resistance, or draw lines, include a DRAW_COMMAND block with JSON:

For multiple drawings at once (preferred):
```DRAW_COMMAND
{{"commands": [
  {{"type": "hline", "price": 595.50, "color": "#26a69a", "label": "Support"}},
  {{"type": "hline", "price": 600.00, "color": "#ef5350", "label": "Resistance"}}
]}}
```

For a single drawing:
```DRAW_COMMAND
{{"type": "hline", "price": 595.50, "color": "#26a69a", "label": "Support"}}
```

Drawing types:
- hline: Horizontal line at a price level
- trendline: Line between two points (requires startTime, startPrice, endTime, endPrice as Unix timestamps)
- zone: Box/rectangle on chart (requires priceHigh, priceLow). Add startTime, endTime (Unix timestamps) to make a bounded box around a specific area. Without times, spans full width. Great for marking patterns (triangles, W-bottoms, consolidation areas).
- clear: Remove all drawings

Example bounded box:
```DRAW_COMMAND
{{"type": "zone", "priceHigh": 610.50, "priceLow": 607.00, "startTime": 1710072000, "endTime": 1710086400, "color": "#2962FF", "label": "Triangle Pattern"}}
```

Colors: #26a69a (green/support), #ef5350 (red/resistance), #FFD700 (gold), #2962FF (blue)

IMPORTANT: Always use triple backticks (```) not single backticks.

Respond naturally and helpfully. If the question is unclear, ask for clarification."""

        try:
            response_text = self._generate(prompt, image_base64=chart_screenshot)
            if chart_screenshot:
                logger.info("Chat response generated with chart screenshot (multimodal)")
            self._log_llm_context('chat_with_context', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in chat: {e}")
            self._log_llm_context('chat_with_context', prompt, error=error_msg)
            return f"Sorry, I encountered an error: {error_msg}"

    async def generate_break_impact(self, critical_items: List[Dict[str, Any]]) -> str:
        """Generate structured break-impact analysis from critical news items."""
        if not self.is_available():
            return "LLM service not available."

        items_text = ""
        for i, item in enumerate(critical_items[:5], 1):
            items_text += (
                f"{i}. [{item.get('impact_tier', 'critical').upper()}] {item.get('headline', '')}\n"
                f"   Sectors: {', '.join(item.get('sector_tags', []))} | "
                f"Sentiment: {item.get('sentiment', 'neutral')} | "
                f"Horizon: {item.get('horizon', 'intraday')}\n"
            )
            reasons = item.get('impact_reasons', [])
            if reasons:
                items_text += f"   Why: {'; '.join(reasons[:3])}\n"
            items_text += "\n"

        prompt = f"""You are a market impact analyst. Analyze the following critical market events and produce a structured break-impact report.

## Critical Events
{items_text}

## Required Output Structure
Produce exactly these sections:

### Why It Matters Now
Brief explanation of why these catalysts are market-moving right now.

### Likely Winners and Losers by Sector
Which sectors benefit and which are at risk. Be specific with sector names and direction.

### What to Monitor Next
Key instruments to watch: mention specific tickers like SPY, QQQ, VIX, UST10Y, DXY, Oil as relevant.

### What Invalidates This View
Specific conditions or events that would negate the current thesis.

Keep the analysis concise, actionable, and tied to the specific events listed above. No generic commentary."""

        try:
            response_text = self._generate(prompt)
            self._log_llm_context('generate_break_impact', prompt, response=response_text)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating break impact: {e}")
            self._log_llm_context('generate_break_impact', prompt, error=error_msg)
            return f"Error generating break impact analysis: {error_msg}"
