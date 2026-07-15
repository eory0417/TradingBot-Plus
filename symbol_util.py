"""심볼 키 정규화 등 공통 헬퍼."""

from __future__ import annotations


def normalize_symbol_key(symbol: str) -> str:
    """ETH/USDT, ETH/USDT:USDT, ETHUSDT → ETH 형태로 비교용 키."""
    return (
        str(symbol)
        .upper()
        .replace(":USDT", "")
        .replace("/USDT", "")
        .replace("USDT", "")
        .replace("/", "")
        .strip()
    )
