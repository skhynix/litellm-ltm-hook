# litellm-ltm-hook

Callback hooks for connecting the LiteLLM proxy to long-term memory (LTM) solutions.

The repository currently provides a hook for [MemMachine](https://github.com/MemMachine/MemMachine). Its broader scope allows hooks for other LTM solutions to be added if needed; no additional integrations are currently planned.

The setup, configuration, and behavior described below apply to the MemMachine hook.

## MemMachine hook

For requests handled by the MemMachine hook:

- **Before** the LLM call: relevant memories for the current user are searched and injected as a `<memory>` system block — a `role:system` message inserted right before the last user message, on all three routes (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`).
- **After** the response: the user's message and the assistant's reply are stored back to the selected MemMachine `(org, project)` namespace.

Clients don't need to change — OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses (`/v1/responses`), and Anthropic Messages (`/v1/messages`) all work. The user's identity is the virtual key owner's email, or — for applications that reach LiteLLM through one shared virtual key — the verified email in Open WebUI's signed user token. Either way it is slugified for `producer`/`produced_for`. Requests identified by a virtual key may select a namespace with `x-memmachine-org` and `x-memmachine-project` (a token-identified request is refused if it sends them); otherwise the slugified user and `default` project are used.

## Files

| Path | Purpose |
|------|---------|
| `memmachine_hook.py` | MemMachine callback hook (single file, no deps beyond `httpx` + `litellm`) |
| `test_hook.py` | Unit + live tests (httpx-mocked parsing/injection/store logic + direct MemMachine API) |
| `test_e2e_store.py` | E2E: store path through the litellm proxy → MemMachine |
| `test_e2e_recall.py` | E2E: 2-turn recall through the litellm proxy → MemMachine |
| `tools/cleanup_memories.py` | Utility: delete all episodic memories in a MemMachine project |
| `DEV.md` | MemMachine hook implementation notes & known limitations |
| `TESTS.md` | MemMachine hook test configuration and usage |

## MemMachine hook setup

### 1. Mount the hook into the LiteLLM container

The hook file needs to be on the LiteLLM proxy's Python import path so the `callbacks` entry can import it. Standard pattern: mount it at `/app/memmachine_hook.py` (LiteLLM's working directory is `/app`).

`docker-compose.yml` (snippet — keep your existing service config):
```yaml
services:
  litellm:
    volumes:
      - ./memmachine_hook.py:/app/memmachine_hook.py:ro
      - ./config.yaml:/app/config.yaml:ro
    command:
      - "--config"
      - "/app/config.yaml"
      - "--port"
      - "4000"
    environment:
      MEMMACHINE_BASE_URL: "http://memmachine.example.com"
      MEMMACHINE_PROJECT_ID: "default"
      MEMMACHINE_AGENT_ID: "litellm"
      MEMMACHINE_TOP_K: "8"
      MEMMACHINE_TIMEOUT: "5"
      MEMMACHINE_SEARCH_TYPES: "episodic"
      MEMMACHINE_MODEL_PREFIX: "memmachine-"
      MEMMACHINE_CLAUDE_CODE_STRIP_SYSTEM_TAGS: "1"
      MEMMACHINE_DEBUG: "0"          # set "1" when debugging
```

### 2. Register the callback in `config.yaml`

```yaml
litellm_settings:
  callbacks:
    - memmachine_hook.memmachine_handler
```

If your existing config already has callbacks, add this entry to the list — don't replace.

### 3. Route only specific models through the memory layer

By default (`MEMMACHINE_MODEL_PREFIX=memmachine-`), only model names starting with `memmachine-` get the hook. To expose a "memory-enabled" variant of a model without duplicating the model definition, add a `model_group_alias` in the proxy's `router_settings` (via config.yaml or the admin UI):

```yaml
router_settings:
  model_group_alias:
    "memmachine-llm-medium": "Qwen/Qwen3.5-122B-A10B-FP8"   # one line per model
```

Clients calling `llm-medium` get the plain model; clients calling `memmachine-llm-medium` get the same backend with the memory hook active.

> Note: the hook detects the prefix **before** the proxy resolves the alias. For store events (`async_log_success_event`), LiteLLM has already resolved the alias (`kwargs.model` is the backend name), so the hook looks up `metadata.model_group` to recover the original requested name. See "Design notes" below.

### 4. Restart LiteLLM

```bash
docker compose up -d
docker logs -f <litellm-container>   # look for "Initialized Callbacks" with MemMachineHook
```

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `MEMMACHINE_BASE_URL` | — (required, no default) | MemMachine server URL. Must be set in `.env` or the real environment; the hook refuses to start without it. |
| `MEMMACHINE_API_KEY` | (none) | Only for deployments requiring auth; leave empty for the default self-hosted setup |
| `MEMMACHINE_PROJECT_ID` | `default` | Default project when `x-memmachine-project` is absent |
| `MEMMACHINE_AGENT_ID` | `litellm` | Used as `producer` for assistant utterances |
| `MEMMACHINE_TOP_K` | `8` | Search result count |
| `MEMMACHINE_TIMEOUT` | `5` | HTTP timeout (seconds) for MemMachine calls |
| `MEMMACHINE_MODELS` | (empty) | Optional comma-separated allowlist; empty = rely on prefix only |
| `MEMMACHINE_MODEL_PREFIX` | `memmachine-` | Only model names with this prefix get the hook; empty string disables the check (applies to all models) |
| `MEMMACHINE_SEARCH_TYPES` | `episodic` | `episodic`, `semantic`, or a comma-separated combination. `all`, `*`, empty, and `[]` request all types; unknown values fail at startup |
| `MEMMACHINE_CLAUDE_CODE_STRIP_SYSTEM_TAGS` | `1` | For `User-Agent: claude-cli/...` requests, strip complete line-leading XML-like wrapper blocks from memory search/store text. Set to `0` to disable; the LLM request itself is never changed. |
| `MEMMACHINE_DEBUG` | (off) | `1` enables verbose stderr diagnostics. Failures always use the standard logger regardless of this setting. |
| `FORWARD_USER_INFO_HEADER_JWT_SECRET` | (none) | The secret Open WebUI signs its user token with — the same variable name Open WebUI uses, because it is the same secret. Unset leaves identity on the virtual key alone. Setting it requires PyJWT. |
| `MEMMACHINE_FORWARDED_USER_JWT_HEADER` | `x-openwebui-user-jwt` | Header carrying that token |
| `MEMMACHINE_FORWARDED_USER_JWT_ISSUER` | `open-webui` | The `iss` claim the token must carry |
| `MEMMACHINE_FORWARDED_USER_JWT_LEEWAY` | `30` | Seconds of clock difference tolerated between Open WebUI and LiteLLM. `0` requires the two clocks to agree exactly |

## How user identity works

`_identity()` accepts an email from one of two sources:

- the authenticated virtual key owner — `user_api_key_dict.user_email` in `async_pre_call_hook`, flattened `user_api_key_user_email` in `async_log_success_event`
- the `email` claim of Open WebUI's signed user token, when `FORWARD_USER_INFO_HEADER_JWT_SECRET` is set and the request carries one

The body `user`, request `metadata.user_id`, `x-memmachine-user`, and virtual-key UUID are not identity fallbacks. An in-scope request without either source fails instead of sharing a `default` identity.

The email is slugified once and the result is the canonical MemMachine user used by `producer` and `produced_for`. It is also the default personal org:

```text
virtual-key email = alice@example.com
user_id           = alice_example_com
org_id            = alice_example_com
project_id        = default
```

The selected `org_id + project_id` is the episodic memory boundary. `producer` is
provenance inside that namespace, not an additional isolation filter. Semantic
searches use MemMachine's default org-set + project-set scope.

Per-request controls:

- `x-memmachine-org: <org>` — select another org; defaults to the slugified user
- `x-memmachine-project: <project>` — select another project; defaults to `MEMMACHINE_PROJECT_ID`
- `x-memmachine-disable: 1` — skip the hook for this request

The hook currently accepts org/project selectors directly for PoC use. A future Authz gateway must authorize the virtual-key-associated MemMachine credential against the requested namespace before forwarding to the raw MemMachine API; see `DEV.md`.

### Identity from Open WebUI

Open WebUI reaches LiteLLM through a single shared virtual key, so the key names
the application and cannot tell one person from the next. Open WebUI can sign a
short-lived token describing the signed-in user and send it with every request:

```bash
# in Open WebUI
ENABLE_FORWARD_USER_INFO_HEADERS=true
FORWARD_USER_INFO_HEADER_JWT_SECRET=<secret>
# in LiteLLM, the same value
FORWARD_USER_INFO_HEADER_JWT_SECRET=<secret>
```

With the secret set, a request carrying a valid token is identified by the token
rather than by the virtual key. The signature, the `iss` claim and the expiry are
all checked, and `HS256` is pinned so a token declaring `alg: none` is refused. A
token that is present but does not verify fails the request with **401**, on
every route and even when the request would not have searched memory; it is
never quietly downgraded to the shared key's identity. This applies to requests
the hook acts on. A request it skips — `x-memmachine-disable: 1`, or a model
outside `MEMMACHINE_MODEL_PREFIX` — is not inspected at all, so a bad token on
one is neither rejected nor able to reach any memory. The store path runs after
the model has answered, so the expiry is judged against the moment the request
started rather than the moment the memory is written.

On a request identified this way, `x-memmachine-org` and `x-memmachine-project`
are refused with **401**, under the same rule as a bad token: even when the
request would not have searched memory. The two rules belong together: behind a shared key the
token is the only thing separating one person's memory from another's, and a
header that re-points the namespace would undo exactly that. This is the
namespace authorization the note above calls for, in the one case where the hook
can decide it alone.

On a request identified by a virtual key the selectors are unchanged, and are
still **not authorized**: a key owned by `alice@example.com` can send
`x-memmachine-org: bob_example_com` and read Bob's memories. That is the PoC
behaviour the note above describes, and this change neither fixes nor worsens
it — an authz gateway that knows which namespaces a credential may reach is
still the answer. What this change does is stop the new identity path from
inheriting the problem, which would otherwise make it pointless.

#### What a deployment has to be true for this to isolate anyone

**The email in the token decides whose memory is used**, and a token-identified
user shares a namespace with a virtual-key user of the same address: both become
`alice_example_com`. That is deliberate — it is what lets one person reach the
same memory from Open WebUI and from a coding agent — but it means Open WebUI's
account system is what the isolation rests on. Open WebUI must not let anyone
register an address they do not own: single sign-on, or verified email, or
signup disabled. With open unverified signup, anyone can register as a
colleague's address and read their memories.

The address is matched exactly, case included: `Alice@Example.com` and
`alice@example.com` become different namespaces. For one person to share memory
across clients, Open WebUI and the virtual key must hold the address in the same
case. The hook does not fold case for you, because doing so would move existing
virtual-key users with mixed-case addresses into new, empty namespaces.

#### Two properties worth knowing

`HS256` is symmetric: the same secret signs and verifies. Anyone who can read
LiteLLM's environment can mint a token for any user, so this is a trust
relationship between two services you operate, not a general-purpose
authentication mechanism.

A token is a bearer credential for that user's memory until it expires, and it
travels as a request header, so it reaches anything LiteLLM passes request
headers to — including logging callbacks such as Langfuse and spend logs. A
token recovered from a log can be replayed for the rest of its lifetime
(Open WebUI's default is five minutes), with any LiteLLM key, not only the one
Open WebUI used. Keep the token lifetime short, and treat logs that capture
request headers accordingly.

## Design notes

Load-bearing implementation details (reasoning-model fallback, `model_group_alias` timing, flattened auth fields, MemMachine API quirks, injection placement, re-injection guard) and current limitations live in [`DEV.md`](./DEV.md). README keeps only what an operator needs to deploy and verify.

## Testing

See [`TESTS.md`](./TESTS.md) for full env-var references and examples. Quick reference:

```bash
# unit + live (requires MEMMACHINE_BASE_URL reachable)
uv run pytest test_hook.py -v
# unit only
MEMMACHINE_LIVE=0 uv run pytest test_hook.py -v

# e2e store path (needs litellm proxy + virtual key + MemMachine up)
LITELLM_MASTER_KEY=sk-... LITELLM_TEST_VIRTUAL_KEY=sk-... \
MEMMACHINE_TEST_EXPECTED_USER=alice@example.com \
python3 test_e2e_store.py

# e2e 2-turn recall
LITELLM_TEST_VIRTUAL_KEY=sk-... python3 test_e2e_recall.py

# wipe all episodic memories in a project (paginated list → delete by uid)
python3 tools/cleanup_memories.py            # --dry-run, --keep USER supported
```

Note: `test_e2e_store.py` verifies storage via `POST /api/v2/memories/list` (not `/search`) — the search endpoint can 500 on edge queries, and list is more reliable for a "did it store?" check.

## Verifying the deployment

After restart, sanity-check in order:

1. **LiteLLM startup log** — `docker logs <litellm>` should show `Initialized Callbacks` containing a `memmachine_hook.MemMachineHook` instance, with no import errors.
2. **2-turn recall** — send a fact, then ask a question that needs it:
   ```bash
   curl http://litellm.example.com/v1/chat/completions \
     -H "Authorization: Bearer <VIRTUAL_KEY>" -H "Content-Type: application/json" \
     -d '{"model":"memmachine-llm-medium","messages":[{"role":"user","content":"I like cats. Remember it."}]}'
   # wait ~5s for indexing
   curl http://litellm.example.com/v1/chat/completions \
     -H "Authorization: Bearer <VIRTUAL_KEY>" -H "Content-Type: application/json" \
     -d '{"model":"memmachine-llm-medium","messages":[{"role":"user","content":"What animal do I like?"}]}'
   ```
   The second response should mention cats.
3. **Logs** — failures are emitted through the standard Python logger even when debug is off. Set `MEMMACHINE_DEBUG=1` temporarily for verbose hook traces such as `async_pre_call_hook`, `_in_scope`, `search ok`, and `store resolved`.

## References

- LiteLLM call hooks: <https://docs.litellm.ai/docs/proxy/call_hooks>
- LiteLLM custom callbacks (`CustomLogger`): <https://docs.litellm.ai/docs/observability/custom_callback>
- MemMachine REST API: <https://docs.memmachine.ai/api_reference>
- MemMachine GitHub: <https://github.com/MemMachine/MemMachine>
- Self-hosted OpenAPI spec: `GET http://memmachine.example.com/openapi.json`

## License

[Apache License 2.0](LICENSE)
