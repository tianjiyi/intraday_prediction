"""
MCP Server for Chart Drawing Tools

This server exposes chart drawing capabilities via the Model Context Protocol (MCP),
allowing LLMs and external tools to draw on the trading chart.

Usage:
    Run standalone: python mcp_chart_server.py
    Or import and use with FastMCP

Tools available:
    - draw_horizontal_line: Draw a horizontal price line
    - draw_trendline: Draw a trendline between two points
    - clear_drawings: Clear all drawings from the chart
    - get_drawings: Get list of current drawings
    - get_current_price: Get the current price of the symbol
"""

import asyncio
import json
import logging
from typing import Optional
from datetime import datetime

# Try to import FastMCP, fall back to simple HTTP server if not available
try:
    from mcp.server.fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    print("FastMCP not installed. Run: pip install mcp")

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chart server configuration
CHART_SERVER_URL = "http://127.0.0.1:5000"
CHART_WS_URL = "ws://127.0.0.1:5000/ws"

# Store for drawings and state
chart_state = {
    "drawings": [],
    "current_price": None,
    "symbol": "QQQ"
}

if HAS_FASTMCP:
    # Create the MCP server
    mcp = FastMCP("chart-drawing-tools")

    @mcp.tool()
    async def draw_horizontal_line(
        price: float,
        color: str = "#FFD700",
        label: Optional[str] = None
    ) -> str:
        """
        Draw a horizontal line at a specific price level on the chart.

        Args:
            price: The price level where the line should be drawn
            color: Line color in hex format (default: gold #FFD700)
            label: Optional label for the line

        Returns:
            Success message with drawing ID or error message
        """
        command = {
            "type": "hline",
            "price": price,
            "color": color,
            "label": label
        }

        result = await send_draw_command(command)
        if result:
            return f"✅ Horizontal line drawn at ${price:.2f}" + (f" ({label})" if label else "")
        return "❌ Failed to draw horizontal line"

    @mcp.tool()
    async def draw_trendline(
        start_time: int,
        start_price: float,
        end_time: int,
        end_price: float,
        color: str = "#2962FF",
        label: Optional[str] = None
    ) -> str:
        """
        Draw a trendline between two points on the chart.

        Args:
            start_time: Unix timestamp for the start point
            start_price: Price at the start point
            end_time: Unix timestamp for the end point
            end_price: Price at the end point
            color: Line color in hex format (default: blue #2962FF)
            label: Optional label for the trendline

        Returns:
            Success message with drawing ID or error message
        """
        command = {
            "type": "trendline",
            "startTime": start_time,
            "startPrice": start_price,
            "endTime": end_time,
            "endPrice": end_price,
            "color": color,
            "label": label
        }

        result = await send_draw_command(command)
        if result:
            return f"✅ Trendline drawn from ${start_price:.2f} to ${end_price:.2f}" + (f" ({label})" if label else "")
        return "❌ Failed to draw trendline"

    @mcp.tool()
    async def draw_support_line(price: float, label: Optional[str] = None) -> str:
        """
        Draw a support line (green) at a specific price level.

        Args:
            price: The support price level
            label: Optional label (default: "Support")

        Returns:
            Success message or error
        """
        return await draw_horizontal_line(price, "#26a69a", label or f"Support ${price:.2f}")

    @mcp.tool()
    async def draw_resistance_line(price: float, label: Optional[str] = None) -> str:
        """
        Draw a resistance line (red) at a specific price level.

        Args:
            price: The resistance price level
            label: Optional label (default: "Resistance")

        Returns:
            Success message or error
        """
        return await draw_horizontal_line(price, "#ef5350", label or f"Resistance ${price:.2f}")

    @mcp.tool()
    async def clear_drawings() -> str:
        """
        Clear all drawings from the chart.

        Returns:
            Confirmation message
        """
        command = {"type": "clear"}
        result = await send_draw_command(command)
        if result:
            return "✅ All drawings cleared from chart"
        return "❌ Failed to clear drawings"

    @mcp.tool()
    async def get_drawings() -> str:
        """
        Get a list of all current drawings on the chart.

        Returns:
            JSON string of current drawings
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{CHART_SERVER_URL}/api/drawings") as response:
                    if response.status == 200:
                        data = await response.json()
                        return json.dumps(data, indent=2)
                    return "❌ Failed to get drawings"
        except Exception as e:
            logger.error(f"Error getting drawings: {e}")
            return f"❌ Error: {str(e)}"

    @mcp.tool()
    async def get_current_price() -> str:
        """
        Get the current price and basic info from the chart.

        Returns:
            Current price information
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{CHART_SERVER_URL}/api/latest_prediction") as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data.get("current_close", "N/A")
                        symbol = data.get("symbol", "N/A")
                        return f"📊 {symbol}: ${price:.2f}" if isinstance(price, (int, float)) else f"📊 {symbol}: {price}"
                    return "❌ Failed to get current price"
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return f"❌ Error: {str(e)}"


async def send_draw_command(command: dict) -> bool:
    """Send a draw command to the chart server via HTTP API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHART_SERVER_URL}/api/draw",
                json=command,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Draw command failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error sending draw command: {e}")
        return False


def main():
    """Run the MCP server."""
    if not HAS_FASTMCP:
        print("Error: FastMCP is required. Install with: pip install mcp")
        return

    print("Starting Chart Drawing MCP Server...")
    print("Available tools:")
    print("  - draw_horizontal_line(price, color, label)")
    print("  - draw_trendline(start_time, start_price, end_time, end_price, color, label)")
    print("  - draw_support_line(price, label)")
    print("  - draw_resistance_line(price, label)")
    print("  - clear_drawings()")
    print("  - get_drawings()")
    print("  - get_current_price()")
    print()

    mcp.run()


if __name__ == "__main__":
    main()
