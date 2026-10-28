"""Delivery: GitHub Issue (full report, works with no extra setup — GitHub
emails/pushes the repo owner) and Telegram (short alert, needs a bot).
Secrets come only from environment variables; nothing is logged."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(text: str, retries: int = 3) -> tuple[bool, str]:
    if not telegram_configured():
        return False, "NOT CONFIGURED (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)"
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]
    for chunk in chunks:
        err = ""
        for _ in range(retries):
            try:
                r = requests.post(url, json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": chunk,
                                             "disable_web_page_preview": True}, timeout=20)
                if r.ok and r.json().get("ok"):
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                err = type(e).__name__
        else:
            return False, err
    return True, "sent"


def github_issue(title: str, body: str, labels: list[str]) -> tuple[bool, str]:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return False, "NOT CONFIGURED (not running in GitHub Actions)"
    mention = os.environ.get("NOTIFY_MENTION")
    if mention:
        body = f"{body}\n\ncc @{mention}"
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          json={"title": title, "body": body[:65000], "labels": labels}, timeout=30)
        if r.status_code == 201:
            return True, r.json()["html_url"]
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, type(e).__name__
