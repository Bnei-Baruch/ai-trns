import json
import os

from tests.conftest import H

from ai_trns.api import requestlog


def entries(api_config):
    with open(os.path.join(api_config.LOG_DIR, requestlog.FILE), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_every_request_is_logged_with_the_caller(client, api_config):
    client.get("/health", headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1", "User-Agent": "probe"})
    client.get("/jobs", headers=H)
    client.get("/jobs")
    health, jobs, rejected = entries(api_config)[-3:]
    assert (health["method"], health["path"], health["status"], health["ip"], health["ua"], health["who"]) == \
           ("GET", "/health", 200, "1.2.3.4", "probe", "-")
    assert (jobs["status"], jobs["who"]) == (200, "tests")
    assert (rejected["status"], rejected["who"]) == (401, "-")


def test_the_proxy_set_ip_wins_over_forwarded_for(client, api_config):
    client.get("/health", headers={"X-Real-IP": "5.6.7.8", "X-Forwarded-For": "6.6.6.6, 5.6.7.8"})
    assert entries(api_config)[-1]["ip"] == "5.6.7.8"


def test_kept_for_log_days(client, api_config):
    client.get("/health")
    assert requestlog._handler.backupCount == api_config.LOG_DAYS == 7
    assert requestlog._handler.when == "MIDNIGHT"
