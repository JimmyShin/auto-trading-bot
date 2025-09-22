# MCP 연동 안내

프로젝트에 Model Context Protocol(MCP)을 추가하여 Claude 등 MCP 클라이언트에서
프로젝트 상태를 안전하게 조회할 수 있습니다.

제공 도구
- `get_state()`: `state.json` 내용을 JSON으로 반환
- `list_csv(kind, limit)`: 최근 `trades_*.csv` 또는 `signal_analysis_*.csv` 목록
- `read_file(path, max_bytes)`: 프로젝트 루트 내 텍스트 파일 읽기(읽기 전용)
- `context7()`: 자동매매 요약 7종 컨텍스트(설정/환경/파일/자산/포지션/상태/노트)
- `get_status()`: 잔고/포지션 요약 + 기본 설정
- `scan_signals(universe, timeframe, lookback)`: 심볼별 신호/레짐/ATR 스냅샷
- `calc_entry(symbol, side, price?)`: 리스크 기반 수량/스탑/리스크 금액 계산
- `sync_state()`: API와 `state.json` 동기화(상태 보정)
- `start_bot()`: 트레이딩 루프 프로세스 실행(보호됨)
- `stop_bot(pid)`: 실행된 루프 중단(보호됨)
- `emergency_close_all(confirm)`: 전체 취소/청산(보호됨)

## 설치

```bash
# 가상환경 활성화 후 (예: .venv)
pip install -r requirements.txt
pip install -r requirements-mcp.txt

# 서버 실행 (stdio 기반)
python mcp_server.py
```

## Claude Desktop 등록 (예시)

Windows 기준 `%APPDATA%\Claude\claude_desktop_config.json` 파일에 다음과 같이 등록하세요.
예시 파일은 `.claude/mcp_servers.example.json` 에도 포함되어 있습니다.

```json
{
  "mcpServers": {
    "donchian-atr-bot": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "C:\\Users\\J\\donchian-atr-bot"
    }
  }
}
```

등록 후 Claude에서 MCP 서버(`donchian-atr-bot`)를 승인하면 위 도구들을 사용할 수 있습니다.

보호 옵션
- 기본값으로 모든 거래 액션은 비활성화되어 있습니다.
- 활성화하려면 환경변수 `ALLOW_MCP_TRADING=true`를 설정하세요.
- 또한 `emergency_close_all`은 `confirm=true` 파라미터가 필요합니다.

추가 요청
- 백테스트/리포트/실행 스케줄링 도구가 필요하시면 알려주세요.
