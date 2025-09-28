from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional


_EMPTY: Dict[str, Dict[str, Dict[str, bool]]] = {"daily": {}, "dedupe": {}}


@dataclass
class StateStore:
    path: Path

    def _load(self) -> Dict:
        if not self.path.exists():
            return {"daily": {}, "dedupe": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"daily": {}, "dedupe": {}}

    def _save(self, data: Dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_daily_peak_equity(self, day: Optional[date] = None) -> Optional[float]:
        day = day or datetime.now(timezone.utc).date()
        data = self._load()
        value = data.get("daily", {}).get(str(day))
        return float(value) if value is not None else None

    def update_daily_peak_equity(self, eq: float, day: Optional[date] = None) -> float:
        day = day or datetime.now(timezone.utc).date()
        data = self._load()
        daily = data.setdefault("daily", {})
        current = float(daily.get(str(day), 0.0) or 0.0)
        updated = max(current, float(eq))
        if updated != current:
            daily[str(day)] = updated
            self._save(data)
        return updated

    def is_deduped_today(self, key: str, day: Optional[date] = None) -> bool:
        day = day or datetime.now(timezone.utc).date()
        data = self._load()
        return str(day) in data.get("dedupe", {}).get(key, {})

    def mark_deduped_today(self, key: str, day: Optional[date] = None) -> None:
        day = day or datetime.now(timezone.utc).date()
        data = self._load()
        dedupe = data.setdefault("dedupe", {}).setdefault(key, {})
        if str(day) not in dedupe:
            dedupe[str(day)] = True
            self._save(data)

