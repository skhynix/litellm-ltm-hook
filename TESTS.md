# Tests for the MemMachine hook

This guide covers the current MemMachine integration in `litellm-ltm-hook`.

Four test files, in increasing scope:

| File | Scope | Requires |
|------|-------|----------|
| `test_hook.py` | unit + live (httpx-mocked + direct MemMachine API) | `httpx`, `pytest`; live needs `MEMMACHINE_BASE_URL` set and reachable |
| `test_identity.py` | unit: the forwarded signed-identity path | `pyjwt`, `pytest`; no network |
| `test_e2e_store.py` | end-to-end store path through the litellm proxy | litellm-test proxy up, virtual key, MemMachine up |
| `test_e2e_recall.py` | end-to-end 2-turn recall through the litellm proxy | same as above |
| `tools/cleanup_memories.py` | utility: delete all episodic memories in a project | MemMachine up |

---

## `test_hook.py` — unit + live (pytest)

Unit tests mock httpx and verify the hook's parsing, injection, and store logic.
Live tests call the configured `MEMMACHINE_BASE_URL` directly (no proxy) to verify the add→search→inject flow.

```bash
# unit + live (requires MEMMACHINE_BASE_URL set and reachable)
uv run pytest test_hook.py -v

# unit only (skip live tests)
MEMMACHINE_LIVE=0 uv run pytest test_hook.py -v
```

---

## `test_e2e_store.py` — end-to-end store path

Verifies that a chat completion through the litellm proxy (authenticated with a virtual key)
gets stored in MemMachine with the virtual-key owner's slugified email as `producer_id`, in that
user's default personal org.

Steps: health check → list models → chat completion (store trigger) → list MemMachine
memories in the user's org → confirm `producer_id` matches the slugified expected email.

### Env vars

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `LITELLM_BASE_URL` | **yes** | — | litellm proxy URL (e.g. `http://litellm.example.com`) |
| `MEMMACHINE_BASE_URL` | **yes** | — | MemMachine URL (e.g. `http://memmachine.example.com`) |
| `LITELLM_MASTER_KEY` | **yes** | — | litellm master key (for `/v1/models`) |
| `LITELLM_TEST_VIRTUAL_KEY` | **yes** | — | virtual key whose owner's email we expect |
| `MEMMACHINE_TEST_MODEL` | no | `memmachine-llm-medium` | model to call |
| `MEMMACHINE_TEST_PROJECT` | no | `default` | MemMachine project_id (shared name across users) |
| `MEMMACHINE_TEST_EXPECTED_USER` | **yes** | — | virtual-key owner email; slugified into the expected `producer_id` and default org_id |

### Run

```bash
LITELLM_MASTER_KEY=sk-... \
LITELLM_TEST_VIRTUAL_KEY=sk-... \
MEMMACHINE_TEST_EXPECTED_USER=alice@example.com \
python3 test_e2e_store.py
```

Exits `0` on success, `1` on failure, `2` on missing env vars.

---

## `test_e2e_recall.py` — end-to-end 2-turn recall

Turn 1: tells the model a fact (stored via `async_log_success_event`).
Turn 2: asks a question that should be answerable from that fact. The hook's
`async_pre_call_hook` searches MemMachine, injects a `<memory>` block, and the
model should recall the fact.

### Env vars

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `LITELLM_BASE_URL` | **yes** | — | litellm proxy URL (e.g. `http://litellm.example.com`) |
| `LITELLM_TEST_VIRTUAL_KEY` | **yes** | — | virtual key for the request |
| `MEMMACHINE_TEST_MODEL` | no | `memmachine-llm-medium` | model to call |
| `MEMMACHINE_TEST_FACT` | no | `my favorite color is purple` | fact to plant in turn 1 |
| `MEMMACHINE_TEST_RECALL_QUERY` | no | `what is my favorite color?` | question in turn 2 |
| `MEMMACHINE_TEST_RECALL_KEYWORD` | no | `purple` | substring expected in turn 2 answer |

### Run

```bash
LITELLM_TEST_VIRTUAL_KEY=sk-... python3 test_e2e_recall.py

# custom fact / query / keyword
LITELLM_TEST_VIRTUAL_KEY=sk-... \
MEMMACHINE_TEST_FACT="I live in Seoul" \
MEMMACHINE_TEST_RECALL_QUERY="where do I live?" \
MEMMACHINE_TEST_RECALL_KEYWORD="Seoul" \
python3 test_e2e_recall.py
```

Exits `0` on success (keyword found in answer), `1` on failure, `2` on missing env vars.

> Note: turn 2 runs 5s after turn 1 to let MemMachine index the new episode.
> If the test fails intermittently, the index may have been slow — just retry.

---

## `tools/cleanup_memories.py` — delete all episodic memories

Lists episodes via `POST /api/v2/memories/list` (paginated, no vector search) and
deletes each by uid. Useful to reset state before/after e2e runs or to wipe a user's org.

The target org_id is determined by (in priority order): `--org`, `--user`, then
the `MEMMACHINE_ORG_ID` env var. With `--user`, the org_id is slugified from the
user identifier the same way the hook does it (e.g. `alice@example.com` ->
`alice_example_com`).

### Env vars

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `MEMMACHINE_BASE_URL` | **yes** | — | MemMachine URL (e.g. `http://memmachine.example.com`) |
| `MEMMACHINE_ORG_ID` | no | — | org to clean (raw org_id; overrides `--user`) |
| `MEMMACHINE_PROJECT_ID` | no | `default` | project to clean |

### Options

- `--org ORG_ID` — target this raw org_id directly (overrides env var)
- `--user USER_ID` — derive org_id from this user identifier (slugified), overrides env var
- `--dry-run` — list what would be deleted; delete nothing
- `--keep USER` — comma-separated canonical (slugified) `producer_id`s to preserve

### Run

```bash
# clean a specific user's org (org_id derived from their email)
python3 tools/cleanup_memories.py --user alice@example.com

# clean a raw org_id directly
python3 tools/cleanup_memories.py --org alice_example_com

# preview what would be deleted
python3 tools/cleanup_memories.py --user alice@example.com --dry-run

# delete all but preserve alice's and bob's memories
python3 tools/cleanup_memories.py --user alice@example.com --keep alice@example.com,bob@example.com
```

---

## Typical workflow

```bash
# 1. reset state (optional, before running e2e) — clean the test user's org
python3 tools/cleanup_memories.py --user alice@example.com

# 2. run e2e store test
LITELLM_MASTER_KEY=sk-... \
LITELLM_TEST_VIRTUAL_KEY=sk-... \
MEMMACHINE_TEST_EXPECTED_USER=alice@example.com \
python3 test_e2e_store.py

# 3. run e2e recall test
LITELLM_TEST_VIRTUAL_KEY=sk-... python3 test_e2e_recall.py

# 4. reset state again (optional)
python3 tools/cleanup_memories.py --user alice@example.com
```

## Notes on the reasoning model

`memmachine-llm-medium` routes to a reasoning model (Qwen3.5-122B), which may
return `content=null` with the answer in `reasoning_content` when `max_tokens`
is too small. The e2e tests use `max_tokens=3000` to ensure the content is fully
generated. If you change the model, adjust `max_tokens` accordingly.
