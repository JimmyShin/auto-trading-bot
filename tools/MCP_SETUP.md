# MCP 설정 안내

이 저장소는 Model Context Protocol(MCP)을 통해 Claude 등 MCP 클라이언트에서 봇 상태를 조회/제어할 수 있는 보조 도구를 제공합니다.

주요 도구
- `get_state()`: `state.json` 내용을 JSON으로 반환
- `list_csv(kind, limit)`: 최근 `trades_*.csv` 또는 `signal_analysis_*.csv` 목록
- `read_file(path, max_bytes)`: 프로젝트 루트 하위 텍스트 파일 읽기(안전한 범위 제한)
- `context7()`: 고수준 7‑파트 컨텍스트(설정/환경/일일/자산/최근 상태/트레이드/시그널)
- `get_status()`: 거래소/자산/설정 요약
- `scan_signals(universe, timeframe, lookback)`: 심볼별 시그널 스냅샷
- `calc_entry(symbol, side, price?)`: 리스크 기반 수량/스탑/금액 계산
- `sync_state()`: API와 `state.json` 상태 보정
- `start_bot()`: 트레이딩 루프 서브프로세스로 실행(보호됨)
- `stop_bot(pid)`: 실행 중인 루프 중단(보호됨)
- `emergency_close_all(confirm)`: 전체 주문 취소 및 포지션 정리(보호됨)

## 설치

```bash
pip install -r requirements.txt
pip install -r requirements-mcp.txt

# 서버 실행 (stdio 기반)
python tools/mcp_server.py
```

## Claude Desktop 등록 (예시)

Windows: `%APPDATA%\Claude\claude_desktop_config.json`에 다음 항목을 추가합니다.

```json
{
  "mcpServers": {
    "donchian-atr-bot": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "cwd": "C:\\path\\to\\repo"
    }
  }
}
```

보호 옵션
- 기본값으로 모든 거래 액션은 비활성화되어 있습니다.
- 활성화하려면 환경변수 `ALLOW_MCP_TRADING=true` 지정
- `emergency_close_all`는 `confirm=true`가 필요합니다.

