import requests
import pandas as pd
from typing import List, Dict, Optional
import os

class PolymarketClient:
    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        # TODO: API key needed for trading, creating markets, and viewing private positions
        # For now, using public market data endpoints only

    def search_markets(self, query: str, limit: int = 100) -> List[Dict]:
        """Search for markets by keyword"""
        endpoint = f"{self.BASE_URL}/markets"
        params = {
            "search_term": query,
            "limit": limit
        }
        try:
            r = self.session.get(endpoint, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            return data.get('data', [])
        except requests.RequestException as e:
            print(f"Search failed: {e}")
            return []

    def get_market(self, market_id: str) -> Optional[Dict]:
        """Get detailed market info"""
        endpoint = f"{self.BASE_URL}/markets/{market_id}"
        try:
            r = self.session.get(endpoint, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Failed to get market {market_id}: {e}")
            return None

    def get_market_prices(self, market_id: str) -> Optional[Dict]:
        """Get current prices for market outcomes"""
        endpoint = f"{self.BASE_URL}/markets/{market_id}/prices"
        try:
            r = self.session.get(endpoint, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Failed to get prices for {market_id}: {e}")
            return None

    def get_order_book(self, market_id: str) -> Optional[Dict]:
        """Get order book for market"""
        endpoint = f"{self.BASE_URL}/markets/{market_id}/orderbook"
        try:
            r = self.session.get(endpoint, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"Failed to get orderbook for {market_id}: {e}")
            return None

    def get_all_markets(self) -> List[Dict]:
        """Get all available markets"""
        endpoint = f"{self.BASE_URL}/markets"
        try:
            r = self.session.get(endpoint, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            return data.get('data', [])
        except requests.RequestException as e:
            print(f"Failed to get all markets: {e}")
            return []

    def filter_biotech_markets(self, markets: List[Dict] = None) -> pd.DataFrame:
        """Filter markets for biotech/clinical trial keywords"""
        if markets is None:
            markets = self.get_all_markets()

        if not markets:
            return pd.DataFrame()

        biotech_keywords = ["gene", "drug", "fda", "trial", "clinical", "pharma", "biotech", "therapy"]
        filtered = [m for m in markets if any(k in m.get('question', '').lower() for k in biotech_keywords)]

        df = pd.DataFrame(filtered)
        return df

    def extract_market_info(self, market: Dict) -> Dict:
        """Extract key info from market data"""
        return {
            "market_id": market.get('id'),
            "question": market.get('question'),
            "outcomes": market.get('outcomes', []),
            "liquidity": market.get('liquidity'),
            "volume_24h": market.get('volume_24h'),
            "created_at": market.get('created_at'),
            "expires_at": market.get('expires_at'),
            "status": market.get('status'),
            "tags": market.get('tags', [])
        }