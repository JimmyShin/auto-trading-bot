from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_alerts_shim() -> bool:
    path = ROOT / "auto_trading_bot" / "alerts.py"
    text = path.read_text(encoding="utf-8")
    shim_line = "import requests as requests  # type: ignore  # noqa: N816"
    if shim_line in text:
        return False
    marker = "from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple\n"
    if marker not in text:
        raise SystemExit("typing import marker not found in alerts.py")
    shim_block = (
        "\n"
        "try:\n"
        "    import requests as requests  # type: ignore  # noqa: N816\n"
        "except Exception:\n"
        "    class _RequestsShim:\n"
        "        def post(self, *a, **k):\n"
        "            raise RuntimeError(\"requests module unavailable\")\n"
        "    requests = _RequestsShim()  # type: ignore\n"
        "\n"
    )
    insert_at = text.index(marker) + len(marker)
    new_text = text[:insert_at] + shim_block + text[insert_at:]
    path.write_text(new_text, encoding="utf-8")
    return True


def _collect_job_block(lines: list[str], start: int) -> tuple[list[str], int]:
    job_lines: list[str] = []
    j = start + 1
    while j < len(lines):
        line = lines[j]
        if line.strip() == "":
            job_lines.append(line)
            j += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2:
            break
        job_lines.append(line)
        j += 1
    return job_lines, j


def _process_job_block(job_lines: list[str]) -> tuple[bool, list[str]]:
    result: list[str] = []
    changed = False
    env_found = False
    i = 0
    while i < len(job_lines):
        line = job_lines[i]
        if line.startswith("    env:"):
            env_found = True
            env_segment: list[str] = ["    env:"]
            content_lines: list[str] = []
            i += 1
            while i < len(job_lines):
                nxt = job_lines[i]
                stripped = nxt.strip()
                indent = len(nxt) - len(nxt.lstrip(" "))
                if stripped == "":
                    content_lines.append(nxt)
                    env_segment.append(nxt)
                    i += 1
                    continue
                if indent >= 6:
                    content_lines.append(nxt)
                    env_segment.append(nxt)
                    i += 1
                    continue
                break
            while i < len(job_lines) and job_lines[i].startswith("    env:"):
                env_segment.append(job_lines[i])
                i += 1
                while i < len(job_lines):
                    nxt = job_lines[i]
                    stripped = nxt.strip()
                    indent = len(nxt) - len(nxt.lstrip(" "))
                    if stripped == "":
                        content_lines.append(nxt)
                        env_segment.append(nxt)
                        i += 1
                        continue
                    if indent >= 6:
                        content_lines.append(nxt)
                        env_segment.append(nxt)
                        i += 1
                        continue
                    break
            new_env_block = ["    env:"]
            py_line = "      PYTHONPATH: ."
            new_env_block.append(py_line)
            seen = {"PYTHONPATH: ."}
            for content in content_lines:
                stripped = content.strip()
                if stripped == "":
                    continue
                if stripped == "PYTHONPATH: .":
                    continue
                if stripped in seen:
                    continue
                new_env_block.append(content)
                seen.add(stripped)
            if env_segment != new_env_block:
                changed = True
            if "PYTHONPATH: ." not in seen:
                changed = True
            result.extend(new_env_block)
            continue
        result.append(line)
        i += 1
    if not env_found:
        new_result: list[str] = []
        inserted = False
        for line in result:
            new_result.append(line)
            if not inserted and line.startswith("    runs-on:"):
                new_result.append("    env:")
                new_result.append("      PYTHONPATH: .")
                inserted = True
        if not inserted:
            raise SystemExit("could not find runs-on line for job env insertion")
        result = new_result
        changed = True
    return changed, result


def ensure_pytest_env() -> bool:
    path = ROOT / ".github" / "workflows" / "pytest.yml"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    result_lines: list[str] = []
    changed = False
    in_jobs = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped == "jobs:":
            in_jobs = True
            result_lines.append(line)
            i += 1
            continue
        if in_jobs and indent == 0 and stripped != "" and stripped != "jobs:" and stripped.endswith(":"):
            in_jobs = False
        if in_jobs and indent == 2 and stripped.endswith(":"):
            job_lines, next_index = _collect_job_block(lines, i)
            job_changed, new_job_lines = _process_job_block(job_lines)
            if job_changed:
                changed = True
            result_lines.append(line)
            result_lines.extend(new_job_lines)
            i = next_index
            continue
        result_lines.append(line)
        i += 1
    if changed:
        new_text = "\n".join(result_lines) + "\n"
        path.write_text(new_text, encoding="utf-8")
    return changed


def main() -> None:
    alerts_changed = ensure_alerts_shim()
    workflow_changed = ensure_pytest_env()
    print(f"alerts_changed={alerts_changed}")
    print(f"workflow_changed={workflow_changed}")


if __name__ == "__main__":
    main()
