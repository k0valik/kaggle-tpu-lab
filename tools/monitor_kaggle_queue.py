#!/usr/bin/env python3
"""Monitor the queue banner in an authenticated Kaggle Draft tab on macOS.

The Kaggle API does not expose an interactive TPU queue position. This script
therefore reads document.body.innerText from a matching, already-open Chrome tab.
Chrome must stay open and "Allow JavaScript from Apple Events" must be enabled.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


QUEUE_RE = re.compile(r"You are\s+#([0-9,]+)\s+in the queue", re.IGNORECASE)
APPLESCRIPT = r'''
on run argv
  set urlNeedle to item 1 of argv
  tell application "Google Chrome"
    repeat with browserWindow in windows
      repeat with browserTab in tabs of browserWindow
        set tabUrl to URL of browserTab
        if tabUrl contains urlNeedle then
          return execute browserTab javascript "document.body.innerText"
        end if
      end repeat
    end repeat
  end tell
  error "No matching Chrome tab" number 44
end run
'''


def stamp(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_config(path):
    data = json.loads(path.read_text())
    required = ("tab_url_contains", "notify_url_template")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError("missing config fields: " + ", ".join(missing))
    if "{content}" not in data["notify_url_template"]:
        raise ValueError("notify_url_template must contain {content}")
    return data


def chrome_tab_text(url_fragment):
    result = subprocess.run(
        ["osascript", "-e", APPLESCRIPT, url_fragment],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def notify(template, message):
    content = urllib.parse.quote(message, safe="")
    url = template.replace("{content}", content)
    request = urllib.request.Request(url, headers={"User-Agent": "kaggle-queue-monitor/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 300:
            raise RuntimeError(f"notification returned HTTP {response.status}")


def parse_position(text):
    match = QUEUE_RE.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def looks_active(text):
    lower = " ".join(text.lower().split())
    if "queued" in lower or "starting" in lower:
        return False
    return any(marker in lower for marker in (
        "draft session running",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config" / "kaggle-tpu-lab" / "queue-monitor.json",
    )
    parser.add_argument("--interval", type=int, default=180, help="poll interval in seconds")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument("--dry-run", action="store_true", help="never send a notification")
    args = parser.parse_args()

    config = load_config(args.config)
    last_position = None
    was_queued = False
    errors = 0

    while True:
        try:
            text = chrome_tab_text(config["tab_url_contains"])
            position = parse_position(text)
            errors = 0

            if position is not None:
                was_queued = True
                if position != last_position:
                    stamp(f"Kaggle TPU queue: #{position}; {max(position - 1, 0)} ahead")
                    last_position = position
                if position <= args.threshold:
                    message = f"Kaggle TPU 排隊到第 {position} 位，前面還有 {max(position - 1, 0)} 人"
                    if not args.dry_run:
                        notify(config["notify_url_template"], message)
                    stamp("threshold reached; notification sent" if not args.dry_run
                          else "threshold reached; dry-run notification skipped")
                    return 0
            elif was_queued and looks_active(text):
                message = "Kaggle TPU Draft Session 已顯示 Running，請查看模型啟動日誌"
                if not args.dry_run:
                    notify(config["notify_url_template"], message)
                stamp("queue cleared; startup notification sent" if not args.dry_run
                      else "queue cleared; dry-run notification skipped")
                return 0
            else:
                stamp("queue position unavailable; Starting alone does not confirm allocation")
        except Exception as exc:
            errors += 1
            stamp(f"check failed ({errors}): {exc}")

        if args.once:
            return 0 if errors == 0 else 1
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        stamp("monitor stopped; Kaggle session is unchanged")
