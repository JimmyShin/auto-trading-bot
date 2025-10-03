from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FakeStateStore:
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@dataclass
class StubExchange:
    account_mode: str = "live"
    set_testnet_calls: List[bool] = field(default_factory=list)

    def set_testnet(self, enabled: bool) -> None:
        self.set_testnet_calls.append(bool(enabled))


@dataclass
class StubNotifier:
    messages: List[str] = field(default_factory=list)

    def send(self, message: str) -> bool:
        self.messages.append(message)
        return True
