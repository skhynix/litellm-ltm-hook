# Copyright 2026 litellm-ltm-hook contributors
# SPDX-License-Identifier: Apache-2.0

"""MemMachine memory layer for the LiteLLM proxy.

async_pre_call_hook       : inject memory search results into messages right before the LLM call
async_log_success_event   : store the conversation after the response completes (incl. streaming)

Supported client routes:
  - OpenAI /v1/chat/completions (call_type "completion"/"acompletion")
  - Anthropic /v1/messages (call_type "anthropic_messages")
  - OpenAI /v1/responses (call_type "aresponses"/"responses")
  LiteLLM normalizes Anthropic messages to OpenAI shape (content as string) and
  /v1/responses `input` to `messages` before the pre-call hook runs, so parsing is
  shared. On all three routes the memory block is injected the same way: a
  role:system entry inserted right before the last user message. The list is
  written back to the same key it was read from (`messages` for Chat/Anthropic,
  `input` for Responses).

config.yaml:
    litellm_settings:
      callbacks: memmachine_hook.memmachine_handler

Environment variables:
  MEMMACHINE_BASE_URL    REQUIRED. No default; the hook refuses to start without it.
  MEMMACHINE_API_KEY     only set for deployments that require auth
  MEMMACHINE_PROJECT_ID  default default
  MEMMACHINE_AGENT_ID    default litellm   (used as producer of assistant utterances)
  MEMMACHINE_TOP_K       default 8
  MEMMACHINE_TIMEOUT     default 5 (seconds)
  MEMMACHINE_MODELS      comma-separated allowlist. Empty = apply to all models.
  MEMMACHINE_MODEL_PREFIX  default "memmachine-". Only model_names starting with this
                         prefix get the memory layer. Empty string disables the prefix
                         check and applies to all (legacy behavior).
  MEMMACHINE_SEARCH_TYPES default episodic. Accepts episodic, semantic, a comma-
                         separated combination, or all/*/empty/[] for all types.
                         Unknown values fail at startup.
  MEMMACHINE_CLAUDE_CODE_STRIP_SYSTEM_TAGS  default 1. For requests whose
                         User-Agent starts with "claude-cli/", remove complete
                         XML-like wrapper blocks from the text used for memory
                         search/store. The original LLM request is not changed.
  MEMMACHINE_DEBUG       1 = log search/store failures to stderr.
  FORWARD_USER_INFO_HEADER_JWT_SECRET  the secret Open WebUI signs its user token
                         with (same name as Open WebUI's own variable). Unset is the
                         default and leaves identity exactly as it was: the virtual
                         key owner's email, with x-memmachine-* namespace selection.
                         Set it, and a request carrying a valid token is identified
                         by the token instead. Requires PyJWT.
  MEMMACHINE_FORWARDED_USER_JWT_HEADER  default x-openwebui-user-jwt
  MEMMACHINE_FORWARDED_USER_JWT_ISSUER  default open-webui. The iss claim the token
                         must carry.

Isolation model:
  Identity is the virtual key owner's email, or the verified email claim of Open
  WebUI's signed token when one is present. The token wins: behind a single shared
  virtual key the key names the application and only the token names the person.
  Either way the email is slugified into the canonical user_id used by
  producer/produced_for and, by default, org_id. alice@example.com maps to
  user_id=alice_example_com, org_id=alice_example_com, project_id=default.

  x-memmachine-org and x-memmachine-project select another namespace. They are
  refused with 401 on a token-identified request, because there the token is the
  only thing separating one person from the next. On a virtual-key request they
  work as before and, as README notes, are still unauthorized: a key owned by
  alice can name bob's namespace. This does not change that either way.

  The email in a token is only as trustworthy as the account system behind Open
  WebUI, since a token-identified user and a virtual-key user with the same email
  share one namespace. See README for what that requires of a deployment.
  Episodic isolation is the selected (org_id, project_id) namespace; producer_id
  records provenance inside that namespace and is not used as a search boundary.
  Semantic searches use MemMachine's default org-set + project-set scope.

MemMachine API (self-hosted, /api/v2 prefix) — verified notes
  POST /api/v2/memories          AddMemoriesSpec { org_id, project_id, types, messages }
  POST /api/v2/memories/search   SearchMemoriesSpec -> SearchResult { status, content }
      content.semantic_memory : [ {category,tag,feature_name,value,...} ]   (empty when semantic is disabled)
      content.episodic_memory : {
          long_term_memory.episodes  : [ {content, producer_id, producer_role, ...} ],
          short_term_memory.episodes : [ {content, ...} ],
          short_term_memory.episode_summary : [ str, ... ]
      }
  - The response payload lives under body["content"], not at the top level.
  - org_id / project_id schema (OpenAPI): letters, digits, _, -, :, and Unicode
    (CJK) only. No @, ., /, or other symbols. The virtual-key email is slugified
    before it is used as the canonical user_id/default org_id (see _to_org_id).
  - The search request deliberately omits filter and set_metadata. Episodic recall
    therefore covers the selected org/project namespace, including all producers.
    Semantic recall uses MemMachine's default org-set + project-set selection. This
    also lets episodic and semantic run together despite their different filter fields.
  - Search 4xx errors are fail-closed except 404 (empty/new namespace), 408
    (timeout), and 429 (temporary availability). Search 5xx/transport failures are
    fail-open. Store failures are always post-response and fail-open, but all failures
    use the standard logger independently of MEMMACHINE_DEBUG.
  - cloud (api.memmachine.ai) uses /v2 prefix + top-level user_id/agent_id fields;
    self-hosted (openapi) uses /api/v2 prefix and has no top-level user_id/agent_id.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import httpx
from litellm.integrations.custom_logger import CustomLogger

try:  # PyJWT. Only needed when the forwarded-identity path is switched on.
    import jwt
except ImportError:  # pragma: no cover - exercised by deployments without PyJWT
    jwt = None


def _load_env_file() -> None:
    """Load KEY=VALUE pairs from a .env file in the hook's directory into os.environ.

    Only sets a variable if it isn't already set in the real environment, so real
    env vars always win. Minimal parser: skips blank lines and comments (#),
    strips an optional leading `export `, and strips surrounding quotes. No
    dependency on python-dotenv — the hook stays single-file.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (len(value) >= 2) and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"[memmachine_hook] .env load failed: {exc!r}", file=sys.stderr, flush=True)


_load_env_file()

_raw_base_url = os.environ.get("MEMMACHINE_BASE_URL", "").rstrip("/")
if not _raw_base_url:
    raise RuntimeError(
        "MEMMACHINE_BASE_URL is required (set it in .env or the real environment)."
    )
BASE_URL = _raw_base_url
API_KEY = os.environ.get("MEMMACHINE_API_KEY")
PROJECT_ID = os.environ.get("MEMMACHINE_PROJECT_ID", "default")
AGENT_ID = os.environ.get("MEMMACHINE_AGENT_ID", "litellm")
TOP_K = int(os.environ.get("MEMMACHINE_TOP_K", "8"))
TIMEOUT = float(os.environ.get("MEMMACHINE_TIMEOUT", "5"))
DEBUG = os.environ.get("MEMMACHINE_DEBUG", "") == "1"
CLAUDE_CODE_STRIP_SYSTEM_TAGS = os.environ.get(
    "MEMMACHINE_CLAUDE_CODE_STRIP_SYSTEM_TAGS", "1"
).strip().lower() in {"1", "true", "yes", "on"}
MODEL_ALLOWLIST = {
    m.strip() for m in os.environ.get("MEMMACHINE_MODELS", "").split(",") if m.strip()
}
MODEL_PREFIX = os.environ.get("MEMMACHINE_MODEL_PREFIX", "memmachine-")

# Forwarded identity. Unset secret = off, and the hook behaves exactly as before.
# The variable deliberately carries Open WebUI's own name: it is the same secret,
# and giving it a second name invites the two sides to drift apart.
FORWARD_USER_INFO_HEADER_JWT_SECRET = os.environ.get(
    "FORWARD_USER_INFO_HEADER_JWT_SECRET"
)
FORWARDED_USER_JWT_HEADER = os.environ.get(
    "MEMMACHINE_FORWARDED_USER_JWT_HEADER", "x-openwebui-user-jwt"
)
FORWARDED_USER_JWT_ISSUER = os.environ.get(
    "MEMMACHINE_FORWARDED_USER_JWT_ISSUER", "open-webui"
)
# Open WebUI stamps iat and exp from its own clock. Two services, two clocks:
# without a margin, an Open WebUI host a couple of seconds ahead makes PyJWT
# raise ImmatureSignatureError on every single request, which reads like a bad
# secret rather than a clock. The margin applies to iat, nbf and exp alike.
# Read only when the feature is on. A malformed value is worth refusing to
# start over, but not for a deployment that never asked for this: with the
# secret unset, nothing here may change how the hook loads.
FORWARDED_USER_JWT_LEEWAY = 30
if FORWARD_USER_INFO_HEADER_JWT_SECRET:
    try:
        FORWARDED_USER_JWT_LEEWAY = max(
            0, int(os.environ.get("MEMMACHINE_FORWARDED_USER_JWT_LEEWAY", "30"))
        )
    except ValueError:
        raise RuntimeError(
            "MEMMACHINE_FORWARDED_USER_JWT_LEEWAY must be a whole number of seconds"
        ) from None

if FORWARD_USER_INFO_HEADER_JWT_SECRET and jwt is not None and not hasattr(
    jwt, "PyJWTError"
):
    # `jwt` is also the name of an unrelated package on PyPI. Importing that one
    # gives a module with a different decode signature, which would fail inside
    # every verification and turn every request into a 401 that looks like a bad
    # token. Say so once, at startup, instead.
    raise RuntimeError(
        "The installed `jwt` module is not PyJWT (it has no PyJWTError). Another "
        "package of that name is shadowing it; install pyjwt and remove the other."
    )

if FORWARD_USER_INFO_HEADER_JWT_SECRET and jwt is None:
    raise RuntimeError(
        "FORWARD_USER_INFO_HEADER_JWT_SECRET is set but PyJWT is not importable. "
        "Install it (pip install pyjwt), or unset the variable to keep identity "
        "on the virtual key alone."
    )

if (
    FORWARD_USER_INFO_HEADER_JWT_SECRET
    and len(FORWARD_USER_INFO_HEADER_JWT_SECRET.encode()) < 32
):
    # HS256 with a short secret can be brute-forced offline from one captured
    # token, and every token is a request header. Warn rather than refuse: an
    # existing deployment should not stop starting because of an upgrade.
    logging.getLogger(f"{__name__}.errors").warning(
        "FORWARD_USER_INFO_HEADER_JWT_SECRET is shorter than 32 bytes. HS256 with "
        "a short secret is brute-forcible offline from a single captured token; "
        "use at least 32 bytes of random data."
    )

_SEARCH_TYPE_VALUES = frozenset({"episodic", "semantic"})


def _parse_search_types(raw: str) -> list[str]:
    """Parse MEMMACHINE_SEARCH_TYPES into the self-hosted API's type list.

    `all`, `*`, an empty string, and the literal `[]` map to the API's empty-list
    representation for all types. A comma-separated episodic/semantic combination
    is accepted, normalized to lowercase, and de-duplicated in input order.
    """
    value = str(raw).strip().lower()
    if value in ("", "all", "*", "[]"):
        return []

    parsed = list(dict.fromkeys(t.strip() for t in value.split(",") if t.strip()))
    invalid = [t for t in parsed if t not in _SEARCH_TYPE_VALUES]
    if invalid:
        allowed = "episodic, semantic, all, or []"
        raise ValueError(
            f"Invalid MEMMACHINE_SEARCH_TYPES value(s): {', '.join(invalid)}; "
            f"allowed values are {allowed}"
        )
    return parsed


SEARCH_TYPES = _parse_search_types(os.environ.get("MEMMACHINE_SEARCH_TYPES", "episodic"))

MEMORY_MARKER = "<memory>"

# Claude Code places synthetic context in role=user as top-level XML-like blocks.
# Match only complete paired blocks that start a line. Repeating the substitution
# also handles adjacent same-line blocks without maintaining a list of tag names.
_CLAUDE_CODE_WRAPPER_BLOCK = re.compile(
    r"(?ms)^[ \t]*<(?P<tag>[A-Za-z][\w:.-]*)(?:\s[^<>]*?)?>"
    r".*?</(?P=tag)\s*>[ \t]*(?:\r?\n)?"
)

_ERROR_LOGGER = logging.getLogger(f"{__name__}.errors")


class MemMachineRequestError(Exception):
    """Fail-closed pre-call error whose HTTP status LiteLLM preserves."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.message = detail
        self.type = _status_category(status_code)
        self.param = None


def _log(msg: str):
    if DEBUG:
        print(f"[memmachine_hook] {msg}", file=sys.stderr, flush=True)


def _request_headers(data: dict) -> dict:
    proxy_request = data.get("proxy_server_request") or {}
    headers = proxy_request.get("headers") or {}
    return headers if isinstance(headers, dict) else {}


def _is_claude_code_request(data: dict) -> bool:
    """Detect Claude Code inference requests without depending on its version."""
    user_agent = _header(_request_headers(data), "user-agent") or ""
    return str(user_agent).strip().lower().startswith("claude-cli/")


def _strip_claude_code_system_tags(text: str) -> str:
    """Remove complete top-level XML-like blocks while preserving plain text.

    Unknown and future tag names are intentionally supported. Incomplete tags are
    preserved so a malformed/user-authored fragment does not consume the rest of the
    prompt. The caller scopes this heuristic to detected Claude Code requests.
    """
    cleaned = text
    while True:
        cleaned, count = _CLAUDE_CODE_WRAPPER_BLOCK.subn("", cleaned)
        if count == 0:
            break
    return cleaned.strip()


def _memory_user_text(data: dict, content) -> str:
    """Extract text for memory operations and apply client-specific cleanup."""
    text = _content_to_text(content)
    if CLAUDE_CODE_STRIP_SYSTEM_TAGS and _is_claude_code_request(data):
        return _strip_claude_code_system_tags(text)
    return text


def _status_category(status_code: int) -> str:
    if status_code == 401:
        return "authentication"
    if status_code == 403:
        return "authorization"
    if status_code in (400, 422):
        return "invalid_request"
    if status_code == 404:
        return "not_found"
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "upstream"
    return "client_error"


def _log_error(
    *,
    operation: str,
    category: str,
    org_id: str,
    project_id: str,
    status_code: int | None = None,
    error_type: str | None = None,
    request_id: str | None = None,
) -> None:
    """Log failures independently of MEMMACHINE_DEBUG without leaking secrets."""
    level = (
        logging.ERROR
        if category
        in {
            "authentication",
            "authorization",
            "invalid_request",
            "client_error",
            "unexpected",
        }
        else logging.WARNING
    )
    _ERROR_LOGGER.log(
        level,
        "MemMachine operation=%s category=%s status=%s org_id=%r project_id=%r "
        "request_id=%r error_type=%s",
        operation,
        category,
        status_code,
        org_id,
        project_id,
        request_id,
        error_type,
    )


def _search_rejection_detail(status_code: int) -> str:
    if status_code == 401:
        return "MemMachine authentication failed"
    if status_code == 403:
        return "MemMachine access denied for the requested memory namespace"
    if status_code in (400, 422):
        return "Invalid MemMachine memory request"
    return "MemMachine rejected the memory request"


def _headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _content_to_text(content) -> str:
    """OpenAI content is either a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _last_user_index(messages) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return None


def _responses_input_to_messages(inp) -> list:
    """Normalize /v1/responses `input` (string or list of {role, content}) to a
    messages list. A bare string becomes a single user message. Returns [] if
    `input` is missing/empty so the caller's _last_user_index check short-circuits.
    """
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}] if inp.strip() else []
    if isinstance(inp, list):
        return [m for m in inp if isinstance(m, dict)]
    return []


def _extract_responses_assistant_text(response_obj) -> str:
    """Extract the assistant's text answer from a /v1/responses response.

    The response's `output` is a list of items; the final answer is in a
    `type="message"` item whose `content` is a list of `{type:"text", text:...}`
    (or `{type:"output_text", text:...}`) blocks. Reasoning items (`type="reasoning"`)
    are skipped — they're the model's chain of thought, not the answer. Returns "" if
    no message item is found, so the caller's non-empty check short-circuits.
    """
    out = getattr(response_obj, "output", None)
    if not isinstance(out, list):
        return ""
    parts: list = []
    for item in out:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type != "message":
            continue
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            text = (
                getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
            )
            if text:
                parts.append(str(text))
    return "\n".join(parts)


# MemMachine's org_id / project_id schema (per OpenAPI) allows: letters, digits,
# underscore, hyphen, colon, and Unicode (CJK). It rejects @, ., /, and other
# symbols. Since user_id is often an email, we slugify it into a valid org_id by
# replacing every disallowed character with an underscore.
_ORG_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_:\-]")


def _to_org_id(user_id: str) -> str:
    """Slugify an identifier using MemMachine's org/project character set.

    The hook uses this for both the canonical user ID and the default personal org.
    Distinct inputs that differ only in disallowed characters can collide; a future
    explicit `memmachine_username` must be validated instead of silently slugified.
    """
    slug = _ORG_ID_ALLOWED.sub("_", str(user_id))
    slug = slug.strip("_")
    return slug or "default"


def _header(headers: dict, name: str):
    """Header lookup that does not depend on the case LiteLLM happened to keep."""
    if name in headers:
        return headers[name]
    want = str(name).lower()
    for key, value in headers.items():
        if str(key).lower() == want:
            return value
    return None


def _unauthorized(message: str) -> MemMachineRequestError:
    """A refusal to trust the caller's identity: 401 on every route it acts on.

    MemMachineRequestError rather than FastAPI's HTTPException, for the reason
    DEV.md gives: LiteLLM keeps its status_code on all three routes, and it does
    not need the proxy extras installed.
    """
    return MemMachineRequestError(status_code=401, detail=message)


def _started_at(start_time) -> Optional[float]:
    """Epoch seconds for a LiteLLM start_time, or None if it is not a datetime.

    The store path runs after the model has answered, which can be well past a
    short-lived token's expiry. Judging the token at that moment would throw away
    the turn the user already had. It is judged against when the request started.
    """
    try:
        return start_time.timestamp()
    except (AttributeError, TypeError, ValueError, OSError):
        return None


def _forwarded_identity(headers: dict, valid_at: Optional[float] = None):
    """The email from Open WebUI's signed user token, or None if there is no token.

    Returns None when the feature is off or no token is present, so the caller
    falls back to the virtual key. Raises when a token is present but not good:
    a request that tried to prove an identity and failed must not be quietly
    downgraded to whatever the shared key says. Also raises when a good token
    arrives with a namespace selector, so every caller that verifies the token
    enforces that rule too, including the pre-call hook's early check.
    """
    if not FORWARD_USER_INFO_HEADER_JWT_SECRET:
        return None
    token = _header(headers, FORWARDED_USER_JWT_HEADER)
    if not token:
        return None

    options = {"require": ["exp", "iss", "email"]}
    if valid_at is not None:
        # Checked below against the request's own start, not against now.
        options["verify_exp"] = False
    try:
        claims = jwt.decode(
            token,
            FORWARD_USER_INFO_HEADER_JWT_SECRET,
            # Pinned: without it a token saying alg=none would be accepted.
            algorithms=["HS256"],
            issuer=FORWARDED_USER_JWT_ISSUER,
            leeway=FORWARDED_USER_JWT_LEEWAY,
            options=options,
        )
    except jwt.PyJWTError as exc:
        # Only PyJWT's own errors mean the token is not good. Anything else — a
        # wrong argument, a shadowed package, a broken install — is ours, and is
        # left to surface as the server error it is rather than being reported
        # to the caller as a bad token.
        raise _unauthorized(
            f"{FORWARDED_USER_JWT_HEADER} could not be verified: {type(exc).__name__}"
        ) from exc

    if valid_at is not None:
        try:
            expires_at = float(claims["exp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _unauthorized(
                f"{FORWARDED_USER_JWT_HEADER} has no usable exp claim"
            ) from exc
        if valid_at > expires_at + FORWARDED_USER_JWT_LEEWAY:
            raise _unauthorized(
                f"{FORWARDED_USER_JWT_HEADER} had already expired when the request "
                "started"
            )

    email = claims.get("email")
    if not email or not str(email).strip():
        raise _unauthorized(
            f"{FORWARDED_USER_JWT_HEADER} carries no email claim"
        )

    # Here the namespace IS the identity. Verifying who the caller is and then
    # letting a header say which memories to read gives back exactly what the
    # signature was introduced to take away: one header, plus a valid token of
    # your own, reads a colleague's memories. Refuse rather than ignore — a
    # caller sending these is either misconfigured or probing, and both should
    # be visible.
    for name in ("x-memmachine-org", "x-memmachine-project"):
        # A blank value means "not selected", as on the virtual-key path.
        if str(_header(headers, name) or "").strip():
            raise _unauthorized(
                f"{name} is not accepted on a request identified by "
                f"{FORWARDED_USER_JWT_HEADER}: the namespace is derived from the "
                "verified user identity"
            )
    return str(email).strip()


def _identity(data, user_api_key_dict=None, valid_at=None):
    """Extract canonical user_id / org_id / project_id / session_id.

    Auth info arrives in two shapes:
    - async_pre_call_hook: user_api_key_dict (a UserAPIKeyAuth object) passed directly.
    - async_log_success_event: flattened user_api_key_* fields inside
      kwargs.litellm_params.metadata (user_api_key_user_email).

    Identity is the virtual key owner's email, or the verified email claim of
    Open WebUI's signed user token when one is present. The token wins: behind a
    single shared virtual key the key names the application and only the token
    names the person. `valid_at` judges the token against the moment the request
    started rather than now, which matters on the store path.

    Returns (user_id, org_id, project_id, session_id) where:
    - user_id is the slugified email used by producer/produced_for.
    - org_id defaults to user_id and may be overridden by x-memmachine-org, except
      on a token-identified request, where the header is refused.
    - project_id defaults to MEMMACHINE_PROJECT_ID, with the same rule.
    - session_id is for session-key scoping (not used for isolation).
    """
    headers = _request_headers(data)
    metadata = data.get("metadata") or {}

    forwarded_email = _forwarded_identity(headers, valid_at)

    user_email = forwarded_email or (
        getattr(user_api_key_dict, "user_email", None)
        or metadata.get("user_api_key_user_email")
    )
    if not user_email or not str(user_email).strip():
        raise ValueError(
            "MemMachine requires a user identity: a user_email on the authenticated "
            f"LiteLLM virtual key, or a verified {FORWARDED_USER_JWT_HEADER} token "
            "(set FORWARD_USER_INFO_HEADER_JWT_SECRET)"
        )

    user_id = _to_org_id(str(user_email).strip())

    if forwarded_email:
        # _forwarded_identity has already refused x-memmachine-org/-project here.
        org_id, project_id = user_id, PROJECT_ID
    else:
        org_id = str(_header(headers, "x-memmachine-org") or user_id).strip()
        project_id = str(_header(headers, "x-memmachine-project") or PROJECT_ID).strip()
    if not org_id or not project_id:
        raise ValueError("MemMachine org_id and project_id must not be empty")
    session_id = (
        _header(headers, "x-memmachine-session")
        or metadata.get("session_id")
        # Behind a shared key the key hash is the same for everyone, so a
        # token-identified request falls back to the person instead.
        or (user_id if forwarded_email else metadata.get("user_api_key_hash"))
        or "default"
    )
    return user_id, org_id, project_id, str(session_id)


def _in_scope(data) -> bool:
    """Whether the memory layer touches this request at all.

    Checked before identity is resolved, so a request the hook skips never has
    its token verified. That costs nothing: skipping means no search and no
    store, so there is no namespace an unverified token could reach.
    """
    if _header(_request_headers(data), "x-memmachine-disable") == "1":
        return False
    model = data.get("model")
    _log(f"_in_scope: model={model!r}, prefix={MODEL_PREFIX!r}, allowlist={MODEL_ALLOWLIST!r}")
    if MODEL_ALLOWLIST and model not in MODEL_ALLOWLIST:
        return False
    # Only apply the memory layer when model_name starts with the prefix.
    # Empty prefix (legacy behavior) applies to all. router_settings.model_group_alias
    # in config.yaml routes "memmachine-X" -> "X", so no memmachine-* entry in
    # model_list is required for this to work.
    if MODEL_PREFIX and not (isinstance(model, str) and model.startswith(MODEL_PREFIX)):
        return False
    return True


def _extract_episodic_texts(episodic) -> list:
    """EpisodicSearchResult -> [str, ...]
    Collect content from long_term/short_term episodes plus short_term episode_summary.
    """
    out: list = []
    if not isinstance(episodic, dict):
        return out
    for term_key in ("long_term_memory", "short_term_memory"):
        term = episodic.get(term_key) or {}
        if not isinstance(term, dict):
            continue
        for ep in term.get("episodes") or []:
            if isinstance(ep, dict):
                text = ep.get("content")
                if text:
                    out.append(str(text))
        # episode_summary exists only under short_term
        for s in term.get("episode_summary") or []:
            if isinstance(s, str) and s.strip():
                out.append(s)
    return out


def _extract_semantic_texts(semantic) -> list:
    """[SemanticFeature] -> [str, ...]. The meaningful value is `value` (or `feature_name`)."""
    out: list = []
    if not isinstance(semantic, list):
        return out
    for feat in semantic:
        if not isinstance(feat, dict):
            continue
        text = feat.get("value") or feat.get("feature_name")
        if text:
            out.append(str(text))
    return out


class MemMachineHook(CustomLogger):
    # ---------- search -> inject ----------
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Supported call types: OpenAI /v1/chat/completions ("completion"/"acompletion"),
        # Anthropic /v1/messages ("anthropic_messages"), and OpenAI /v1/responses
        # ("aresponses"/"responses"). LiteLLM normalizes Anthropic messages to OpenAI
        # shape (content as string) before this hook runs; /v1/responses keeps `input`
        # (string or list of {role,content}) and `instructions` (top-level system prompt).
        if call_type not in (
            "completion", "acompletion",
            "anthropic_messages", "aanthropic_messages",
            "aresponses", "responses",
        ):
            return data
        _log(f"async_pre_call_hook: data.model={data.get('model')!r}, "
             f"call_type={call_type!r}, "
             f"user_email={getattr(user_api_key_dict,'user_email',None)!r}, "
             f"user_id={getattr(user_api_key_dict,'user_id',None)!r}")
        if not _in_scope(data):
            return data

        # Verify a forwarded token before any of the early returns below, so a
        # forged or expired one, or a good one sent with a namespace selector, is
        # refused even on a request that would not have searched memory.
        # _identity checks it again; that is cheap.
        _forwarded_identity(_request_headers(data))

        # /v1/responses carries the conversation as `input` (string or list of
        # {role, content}) rather than `messages`. Normalize to a messages list so the
        # shared logic below works; the original data["input"] is never mutated.
        is_responses = call_type in ("aresponses", "responses")
        if is_responses:
            messages = _responses_input_to_messages(data.get("input"))
        else:
            messages = data.get("messages") or []
        idx = _last_user_index(messages)
        if idx is None:
            return data

        query = _memory_user_text(data, messages[idx].get("content"))
        if not query.strip():
            return data

        # If a <memory> block was already injected in a previous turn, do not re-inject.
        # (See DEV.md section 6: each turn is an ephemeral branch. The guard only fires
        # when the client itself sends back a history already containing <memory>.)
        if any(
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and MEMORY_MARKER in m["content"]
            for m in messages
        ):
            _log("pre_call: <memory> block already in history, skipping re-injection")
            return data

        # No valid_at here: this runs before the model is called, so "now" is
        # the moment the request arrived. The store path runs after the answer
        # and has to look back at when the request started instead.
        _, org_id, project_id, _ = _identity(data, user_api_key_dict)

        # Namespace selection, not producer_id, defines the episodic recall boundary.
        # Omitting filter/set_metadata also asks semantic memory to use its default
        # org-set + project-set scope and keeps mixed/all searches compatible.
        request_id = data.get("litellm_call_id")

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    f"{BASE_URL}/api/v2/memories/search",
                    json={
                        "org_id": org_id,
                        "project_id": project_id,
                        "query": query,
                        "top_k": TOP_K,
                        "types": SEARCH_TYPES,
                    },
                    headers=_headers(),
                )
                resp.raise_for_status()
                body = resp.json()
                if not isinstance(body, dict):
                    raise ValueError("MemMachine search response must be a JSON object")
                content = body.get("content") or {}
                if not isinstance(content, dict):
                    raise ValueError("MemMachine search content must be a JSON object")
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            category = _status_category(status_code)
            _log_error(
                operation="search",
                category=category,
                status_code=status_code,
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            if 400 <= status_code < 500 and status_code not in (404, 408, 429):
                raise MemMachineRequestError(
                    status_code=status_code,
                    detail=_search_rejection_detail(status_code),
                ) from None
            return data
        except httpx.TimeoutException as exc:
            _log_error(
                operation="search",
                category="timeout",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return data
        except httpx.TransportError as exc:
            _log_error(
                operation="search",
                category="transport",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return data
        except Exception as exc:
            _log_error(
                operation="search",
                category="unexpected",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return data

        episodic = content.get("episodic_memory")
        semantic = content.get("semantic_memory")

        lines = _extract_semantic_texts(semantic) + _extract_episodic_texts(episodic)
        _log(f"search ok: org_id={org_id!r}, project_id={project_id!r}, "
             f"query={query!r}, types={SEARCH_TYPES!r}, "
             f"lines={len(lines)}, raw_body_keys={list(body.keys())}, "
             f"has_content={'content' in body}")
        if not lines:
            _log("search returned no lines — skipping injection")
            return data

        block = (
            f"{MEMORY_MARKER}\n"
            "Below is information retrieved from previous sessions in the selected "
            "memory context. Use it only when relevant; do not assume every item "
            "describes the current user, and do not fabricate anything not shown here.\n"
            + "\n".join(f"- {line}" for line in lines)
            + "\n</memory>"
        )
        # All three routes: insert a role:system block right before the last user
        # message. Putting it at the front would change the prompt prefix every turn
        # and invalidate the vLLM prefix cache for the whole prompt. On /v1/responses
        # the list lives under `input` (not `messages`), so write back to the same key
        # we read from above.
        messages.insert(idx, {"role": "system", "content": block})
        if is_responses:
            data["input"] = messages
        else:
            data["messages"] = messages
        return data

    # ---------- store ----------
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        litellm_params = kwargs.get("litellm_params") or {}
        data = litellm_params.get("metadata") or {}
        # Anthropic /v1/messages route keeps the same flattened auth/model fields under
        # litellm_params.litellm_metadata instead of litellm_params.metadata. Fall back
        # to it so store + _identity work on both routes.
        litellm_metadata = litellm_params.get("litellm_metadata") or {}
        meta = {**litellm_metadata, **data}  # data wins on overlap (OpenAI path)
        request_id = (
            kwargs.get("litellm_call_id")
            or litellm_params.get("litellm_call_id")
            or meta.get("litellm_call_id")
        )
        _log(f"async_log_success_event: kwargs.model={kwargs.get('model')!r}, "
             f"metadata.model_group={data.get('model_group')!r}, "
             f"litellm_metadata.model_group={litellm_metadata.get('model_group')!r}")

        # Recover the original requested model name. kwargs.model is the value AFTER
        # the router resolves model_group_alias (e.g. "memmachine-llm-medium" ->
        # "Qwen/Qwen3.5-..."). The original name is preserved in:
        #   - OpenAI route:  litellm_params.metadata.model_group
        #   - Anthropic route: litellm_params.litellm_metadata.model_group
        #   - both routes:   litellm_params.proxy_server_request.body.model
        psr = litellm_params.get("proxy_server_request") or data.get("proxy_server_request") or {}
        psr_body = psr.get("body") if isinstance(psr, dict) else None
        psr_body_model = psr_body.get("model") if isinstance(psr_body, dict) else None
        model_name = (
            data.get("model_group")
            or litellm_metadata.get("model_group")
            or psr_body_model
            or data.get("model")
            or kwargs.get("model")
        )
        request = {
            "proxy_server_request": psr,
            "metadata": meta,
            "model": model_name,
            "user": kwargs.get("user"),
        }
        if not _in_scope(request):
            _log("async_log_success_event: _in_scope False, skipping store")
            return

        # /v1/responses carries the conversation as `input` (string or list of
        # {role, content}). LiteLLM normalizes it to `kwargs["messages"]` (list of
        # {role, content}), so the shared _last_user_index / _content_to_text work.
        messages = kwargs.get("messages") or []
        idx = _last_user_index(messages)
        if idx is None:
            return
        user_text = _memory_user_text(request, messages[idx].get("content"))

        call_type = kwargs.get("call_type")
        is_responses = call_type in ("aresponses", "responses")
        try:
            if is_responses:
                assistant_text = _extract_responses_assistant_text(response_obj)
            else:
                choices = getattr(response_obj, "choices", None) or response_obj["choices"]
                message = choices[0].message if hasattr(choices[0], "message") else choices[0]["message"]
                # Reasoning models (e.g. Kimi-K3, Qwen3.5) may return content=null and put
                # the answer in reasoning_content. Prefer content; fall back to reasoning.
                msg_content = getattr(message, "content", None)
                if msg_content is None and isinstance(message, dict):
                    msg_content = message.get("content")
                assistant_text = _content_to_text(msg_content)
                if not assistant_text.strip():
                    reasoning = (
                        getattr(message, "reasoning_content", None)
                        or (message.get("reasoning_content") if isinstance(message, dict) else None)
                        or getattr(message, "reasoning", None)
                        or (message.get("reasoning") if isinstance(message, dict) else None)
                    )
                    assistant_text = _content_to_text(reasoning)
        except Exception as exc:
            _log_error(
                operation="store",
                category="unexpected",
                org_id="<unresolved>",
                project_id="<unresolved>",
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return

        if not (user_text.strip() and assistant_text.strip()):
            return

        # async_log_success_event does not receive user_api_key_dict directly.
        # Auth info lives in kwargs.litellm_params.metadata (OpenAI) or
        # litellm_params.litellm_metadata (Anthropic); both are already merged into
        # `meta` above, and passed to _identity via request["metadata"].
        user_api_key_dict = (
            kwargs.get("user_api_key_dict")
            or litellm_params.get("user_api_key_dict")
            or None
        )
        _log(f"store identity: user_api_key_dict type={type(user_api_key_dict).__name__}, "
             f"kwargs.user={kwargs.get('user')!r}, "
             f"meta keys={list(meta.keys())[:30]}, "
             f"meta.user_api_key_dict={meta.get('user_api_key_dict')!r}")
        # If a user_api_key_dict is embedded in metadata, use it.
        if user_api_key_dict is None and isinstance(meta.get("user_api_key_dict"), dict):
            from litellm.proxy._types import UserAPIKeyAuth
            try:
                user_api_key_dict = UserAPIKeyAuth(**meta["user_api_key_dict"])
            except Exception as exc:
                _log_error(
                    operation="store",
                    category="unexpected",
                    org_id="<unresolved>",
                    project_id="<unresolved>",
                    request_id=request_id,
                    error_type=type(exc).__name__,
                )
        try:
            user_id, org_id, project_id, session_id = _identity(
                request, user_api_key_dict, valid_at=_started_at(start_time)
            )
        except Exception as exc:
            _log_error(
                operation="store",
                # A wrong secret or issuer otherwise looks like any other
                # failure, which is a miserable thing to diagnose from logs.
                category=(
                    "authentication"
                    if isinstance(exc, MemMachineRequestError)
                    else "unexpected"
                ),
                org_id="<unresolved>",
                project_id="<unresolved>",
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return
        _log(f"store resolved: user_id={user_id!r}, org_id={org_id!r}, "
             f"project_id={project_id!r}, session_id={session_id!r}")
        meta = {
            "user_id": user_id,
            "agent_id": AGENT_ID,
            "session_id": session_id,
            "group_id": "default",
        }
        payload = {
            "org_id": org_id,
            "project_id": project_id,
            "types": [],
            "messages": [
                {
                    "content": user_text,
                    # producer records who created the episode inside the selected
                    # org/project namespace; it is provenance, not an isolation key.
                    "producer": user_id,
                    "produced_for": AGENT_ID,
                    "role": "user",
                    "metadata": meta,
                },
                {
                    "content": assistant_text,
                    "producer": AGENT_ID,
                    "produced_for": user_id,
                    "role": "assistant",
                    "metadata": meta,
                },
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    f"{BASE_URL}/api/v2/memories",
                    json=payload,
                    headers=_headers(),
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            _log_error(
                operation="store",
                category=_status_category(status_code),
                status_code=status_code,
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
        except httpx.TimeoutException as exc:
            _log_error(
                operation="store",
                category="timeout",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
        except httpx.TransportError as exc:
            _log_error(
                operation="store",
                category="transport",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            _log_error(
                operation="store",
                category="unexpected",
                org_id=org_id,
                project_id=project_id,
                request_id=request_id,
                error_type=type(exc).__name__,
            )


memmachine_handler = MemMachineHook()
