"""网页端口（83/84/85）的薄封装：统一把 400 的中文 message 变成 DuckCommandError。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests

from .errors import DuckCommandError

if TYPE_CHECKING:
    from . import Duck


def url(duck: Duck, page: str, path: str) -> str:
    return f"http://{duck.host}:{duck.web_ports[page]}{path}"


def _json_or_text(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _raise_for(response: requests.Response, body: dict[str, Any]) -> None:
    if response.status_code < 400 and body.get("ok") is not False:
        return
    reason = body.get("message") or f"HTTP {response.status_code}: {response.text[:200]}"
    raise DuckCommandError(reason, status_code=response.status_code)


def get_json(duck: Duck, page: str, path: str, *, params: dict | None = None,
             timeout: float = 10.0) -> dict[str, Any]:
    response = requests.get(url(duck, page, path), params=params, timeout=timeout)
    body = _json_or_text(response)
    _raise_for(response, body)
    return body


def post_json(duck: Duck, page: str, path: str, *, params: dict | None = None,
              data: bytes | None = None, headers: dict | None = None,
              timeout: float = 30.0) -> dict[str, Any]:
    response = requests.post(url(duck, page, path), params=params, data=data,
                             headers=headers, timeout=timeout)
    body = _json_or_text(response)
    _raise_for(response, body)
    return body


def get_stream(duck: Duck, page: str, path: str, *, params: dict | None = None,
               timeout: float = 60.0) -> requests.Response:
    response = requests.get(url(duck, page, path), params=params, timeout=timeout, stream=True)
    if response.status_code >= 400:
        body = _json_or_text(response)
        reason = body.get("message") or f"HTTP {response.status_code}: {response.text[:200]}"
        response.close()
        raise DuckCommandError(reason, status_code=response.status_code)
    return response
