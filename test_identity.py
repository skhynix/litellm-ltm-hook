# Copyright 2026 litellm-ltm-hook contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the forwarded signed-identity path.

These cover _forwarded_identity and the parts of _identity that depend on it.
Nothing here makes a network call.

    uv run pytest test_identity.py -v
"""
import asyncio
import datetime
import os
import time

# Same placeholder the rest of the suite uses: the hook needs a base URL at
# import time, and nothing here makes a real request.
os.environ.setdefault("MEMMACHINE_BASE_URL", "http://memmachine.unit.test")

import pytest

import memmachine_hook as h

jwt = pytest.importorskip("jwt", reason="PyJWT is needed for the identity path")

# 32+ bytes, so PyJWT does not warn about HMAC key length in test output.
SECRET = "test-secret-not-a-real-one-0123456789"
KEY_EMAIL = "llm-chat-service@example.com"
PERSON = "alice@example.com"


@pytest.fixture
def signing(monkeypatch):
    """Turn the forwarded-identity path on, as a deployment would."""
    monkeypatch.setattr(h, "FORWARD_USER_INFO_HEADER_JWT_SECRET", SECRET)
    monkeypatch.setattr(h, "FORWARDED_USER_JWT_HEADER", "x-openwebui-user-jwt")
    monkeypatch.setattr(h, "FORWARDED_USER_JWT_ISSUER", "open-webui")
    return SECRET


def token(email=PERSON, *, secret=SECRET, issuer="open-webui", ttl=300,
          iat_offset=0, algorithm="HS256", drop=()):
    """A token shaped like the one Open WebUI mints."""
    now = int(time.time()) + iat_offset
    claims = {
        "sub": "7b1c2d3e-0000-4444-8888-999999999999",
        "email": email,
        "name": "Alice Example",
        "role": "user",
        "iss": issuer,
        "iat": now,
        "exp": now + ttl,
    }
    for name in drop:
        claims.pop(name, None)
    key = "" if algorithm == "none" else secret
    return jwt.encode(claims, key, algorithm=algorithm)


def request(headers=None, *, key_email=KEY_EMAIL):
    """A pre-call `data` dict with an authenticated virtual key behind it."""
    return (
        {"proxy_server_request": {"headers": dict(headers or {})}},
        type("Key", (), {"user_email": key_email})(),
    )


def status_of(exc):
    """The HTTP status LiteLLM will return for this refusal."""
    return getattr(exc, "status_code", None)


class TestOffByDefault:
    def test_no_secret_keeps_the_virtual_key_identity(self):
        data, key = request({"x-openwebui-user-jwt": token()})
        user_id, org_id, project_id, _ = h._identity(data, key)
        assert user_id == h._to_org_id(KEY_EMAIL)
        assert org_id == h._to_org_id(KEY_EMAIL)
        assert project_id == h.PROJECT_ID

    def test_no_secret_still_honours_namespace_headers(self):
        data, key = request({"x-memmachine-org": "shared", "x-memmachine-project": "team"})
        _, org_id, project_id, _ = h._identity(data, key)
        assert (org_id, project_id) == ("shared", "team")

    def test_secret_set_but_no_token_falls_back_to_the_key(self, signing):
        data, key = request()
        user_id, org_id, _, _ = h._identity(data, key)
        assert user_id == org_id == h._to_org_id(KEY_EMAIL)


class TestTokenWins:
    def test_token_identity_beats_the_virtual_key(self, signing):
        data, key = request({"x-openwebui-user-jwt": token()})
        user_id, org_id, project_id, _ = h._identity(data, key)
        assert user_id == org_id == h._to_org_id(PERSON)
        assert project_id == h.PROJECT_ID

    def test_two_people_behind_one_shared_key_stay_apart(self, signing):
        first, key = request({"x-openwebui-user-jwt": token("alice@example.com")})
        second, _ = request({"x-openwebui-user-jwt": token("bob@example.com")})
        assert h._identity(first, key)[1] != h._identity(second, key)[1]

    def test_header_name_is_case_insensitive(self, signing):
        data, key = request({"X-OpenWebUI-User-Jwt": token()})
        assert h._identity(data, key)[0] == h._to_org_id(PERSON)

    def test_a_token_alone_is_enough(self, signing):
        """No virtual key email at all: the shared-key deployment's normal case."""
        data, _ = request({"x-openwebui-user-jwt": token()})
        assert h._identity(data, None)[0] == h._to_org_id(PERSON)


class TestBadTokensAreRefused:
    """A request that tried to prove an identity and failed is never downgraded
    to whatever the shared key says."""

    @pytest.mark.parametrize(
        ("name", "bad"),
        [
            ("wrong signature", lambda: token(secret="a-different-secret-0123456789abcdef")),
            ("alg none", lambda: token(algorithm="none")),
            ("wrong issuer", lambda: token(issuer="somebody-else")),
            ("already expired", lambda: token(ttl=-60)),
            ("no exp claim", lambda: token(drop=("exp",))),
            ("no iss claim", lambda: token(drop=("iss",))),
            ("not a token", lambda: "nonsense.not.ajwt"),
        ],
    )
    def test_refused(self, signing, name, bad):
        data, key = request({"x-openwebui-user-jwt": bad()})
        with pytest.raises(Exception) as caught:
            h._identity(data, key)
        assert status_of(caught.value) == 401

    def test_no_email_claim_is_refused(self, signing):
        """Caught by the required-claims list, before anything looks at the value."""
        data, key = request({"x-openwebui-user-jwt": token(drop=("email",))})
        with pytest.raises(Exception) as caught:
            h._identity(data, key)
        assert status_of(caught.value) == 401

    def test_blank_email_claim_is_refused(self, signing):
        """Present but empty: the required-claims list is satisfied, the value is not."""
        data, key = request({"x-openwebui-user-jwt": token(email="   ")})
        with pytest.raises(Exception) as caught:
            h._identity(data, key)
        assert "email" in str(getattr(caught.value, "detail", caught.value))

    def test_a_refusal_is_the_hooks_own_error(self, signing):
        """MemMachineRequestError keeps its 401 on every route (see DEV.md), and
        lets the store path log it as an authentication failure."""
        data, key = request({"x-openwebui-user-jwt": token(secret="wrong-but-long-enough-0123456789abcdef")})
        with pytest.raises(h.MemMachineRequestError) as caught:
            h._identity(data, key)
        assert caught.value.status_code == 401


class TestNamespaceHeaders:
    """The guard that makes the identity mean something.

    Verifying who the caller is and then letting a header choose whose memories
    to read would undo the verification. With a virtual key per user there is no
    such problem, so the headers keep working there.
    """

    @pytest.mark.parametrize("header", ["x-memmachine-org", "x-memmachine-project"])
    def test_refused_on_a_token_identified_request(self, signing, header):
        data, key = request({"x-openwebui-user-jwt": token(), header: "someone-else"})
        with pytest.raises(Exception) as caught:
            h._identity(data, key)
        assert status_of(caught.value) == 401
        assert header in str(getattr(caught.value, "detail", caught.value))

    def test_refused_whatever_the_case(self, signing):
        data, key = request({"x-openwebui-user-jwt": token(), "X-MemMachine-Org": "x"})
        with pytest.raises(Exception):
            h._identity(data, key)

    @pytest.mark.parametrize("header", ["x-memmachine-org", "x-memmachine-project"])
    def test_an_empty_value_counts_as_absent(self, signing, header):
        """As on the virtual-key path, where an empty value falls back to the default."""
        data, key = request({"x-openwebui-user-jwt": token(), header: ""})
        _, org_id, project_id, _ = h._identity(data, key)
        assert (org_id, project_id) == (h._to_org_id(PERSON), h.PROJECT_ID)

    @pytest.mark.parametrize("header", ["x-memmachine-org", "x-memmachine-project"])
    def test_a_blank_value_counts_as_absent(self, signing, header):
        """Whitespace selects nothing, so it is not a selector to refuse."""
        data, key = request({"x-openwebui-user-jwt": token(), header: "   "})
        _, org_id, project_id, _ = h._identity(data, key)
        assert (org_id, project_id) == (h._to_org_id(PERSON), h.PROJECT_ID)

    def test_still_allowed_without_a_token(self, signing):
        data, key = request({"x-memmachine-org": "shared"})
        assert h._identity(data, key)[1] == "shared"

    def test_session_header_is_untouched(self, signing):
        data, key = request({"x-openwebui-user-jwt": token(), "x-memmachine-session": "s1"})
        assert h._identity(data, key)[3] == "s1"


class TestSessionFallback:
    """Behind a shared key the key hash is the same for everyone."""

    def test_token_request_falls_back_to_the_person(self, signing):
        data, key = request({"x-openwebui-user-jwt": token()})
        data["metadata"] = {"user_api_key_hash": "shared-key-hash"}
        assert h._identity(data, key)[3] == h._to_org_id(PERSON)

    def test_key_request_still_falls_back_to_the_key_hash(self, signing):
        data, key = request()
        data["metadata"] = {"user_api_key_hash": "personal-key-hash"}
        assert h._identity(data, key)[3] == "personal-key-hash"


class TestStorePathExpiry:
    """The store path runs after the model has answered, which can be past a
    short-lived token's expiry. It is judged against when the request started."""

    def test_token_that_expired_during_the_call_is_still_accepted(self, signing):
        started = time.time() - 600
        data, key = request({"x-openwebui-user-jwt": token(ttl=300, iat_offset=-600)})
        assert h._identity(data, key, valid_at=started)[0] == h._to_org_id(PERSON)

    def test_token_already_expired_when_the_request_started_is_refused(self, signing):
        started = time.time()
        data, key = request({"x-openwebui-user-jwt": token(ttl=300, iat_offset=-3600)})
        with pytest.raises(Exception) as caught:
            h._identity(data, key, valid_at=started)
        assert status_of(caught.value) == 401

    def test_a_forged_token_is_refused_on_the_store_path_too(self, signing):
        data, key = request({"x-openwebui-user-jwt": token(secret="wrong-but-long-enough-0123456789abcdef")})
        with pytest.raises(Exception):
            h._identity(data, key, valid_at=time.time())

    def test_started_at_tolerates_a_missing_datetime(self):
        assert h._started_at(None) is None
        assert h._started_at("not a datetime") is None

    def test_without_a_start_time_a_long_call_loses_its_store(self, signing):
        """What the fallback costs, written down.

        _started_at returning None means valid_at is None, so expiry falls back
        to now. A token that expired while the model was answering is then
        refused and the turn is not stored. The turn the user already had is
        lost rather than written to the wrong namespace, which is the right way
        round, but it is a real cost of not knowing when the request started.
        """
        data, key = request({"x-openwebui-user-jwt": token(ttl=300, iat_offset=-3600)})
        assert h._started_at(None) is None
        with pytest.raises(Exception) as caught:
            h._identity(data, key, valid_at=h._started_at(None))
        assert status_of(caught.value) == 401


class TestClockSkew:
    """Open WebUI and LiteLLM stamp and read the token on different clocks.

    PyJWT refuses a token whose iat is even a second in the future, so without a
    margin an Open WebUI host slightly ahead fails every request, with an error
    that reads like a bad secret rather than a clock.
    """

    @pytest.mark.parametrize("skew", [1, 5, 29])
    def test_a_token_from_a_clock_that_is_ahead_is_accepted(self, signing, skew):
        data, key = request({"x-openwebui-user-jwt": token(iat_offset=skew)})
        assert h._identity(data, key)[0] == h._to_org_id(PERSON)

    def test_a_clock_further_ahead_than_the_margin_is_refused(self, signing, monkeypatch):
        monkeypatch.setattr(h, "FORWARDED_USER_JWT_LEEWAY", 5)
        data, key = request({"x-openwebui-user-jwt": token(iat_offset=120)})
        with pytest.raises(Exception) as caught:
            h._identity(data, key)
        assert status_of(caught.value) == 401

    def test_the_margin_also_covers_expiry(self, signing, monkeypatch):
        monkeypatch.setattr(h, "FORWARDED_USER_JWT_LEEWAY", 30)
        # Expired 10 seconds ago: inside the margin, so still accepted.
        data, key = request({"x-openwebui-user-jwt": token(ttl=-10)})
        assert h._identity(data, key)[0] == h._to_org_id(PERSON)

    def test_zero_margin_restores_strict_checking(self, signing, monkeypatch):
        monkeypatch.setattr(h, "FORWARDED_USER_JWT_LEEWAY", 0)
        data, key = request({"x-openwebui-user-jwt": token(ttl=-10)})
        with pytest.raises(Exception):
            h._identity(data, key)


class TestRefusalIsA401:
    def test_bad_token(self, signing):
        data, key = request({"x-openwebui-user-jwt": token(secret="wrong-but-long-enough-0123456789abcdef")})
        with pytest.raises(h.MemMachineRequestError) as caught:
            h._identity(data, key)
        assert caught.value.status_code == 401

    def test_namespace_header(self, signing):
        data, key = request({"x-openwebui-user-jwt": token(), "x-memmachine-org": "x"})
        with pytest.raises(h.MemMachineRequestError) as caught:
            h._identity(data, key)
        assert caught.value.status_code == 401


FORGED = {"x-openwebui-user-jwt": "nonsense.not.ajwt"}


class TestPreCallRefusesBeforeEarlyReturns:
    """The pre-call hook skips memory for some requests before it resolves the
    identity. A bad token on those must still be refused, not passed to the model."""

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([{"role": "assistant", "content": "hi"}], id="no user message"),
            pytest.param([{"role": "user", "content": "   "}], id="empty query"),
            pytest.param(
                [
                    {"role": "system", "content": h.MEMORY_MARKER + " earlier"},
                    {"role": "user", "content": "hi"},
                ],
                id="memory already injected",
            ),
        ],
    )
    def test_bad_token_is_refused(self, signing, messages):
        data = {
            "model": "memmachine-m",
            "messages": messages,
            "proxy_server_request": {"headers": dict(FORGED)},
        }
        with pytest.raises(h.MemMachineRequestError) as caught:
            asyncio.run(h.MemMachineHook().async_pre_call_hook(None, None, data, "completion"))
        assert caught.value.status_code == 401

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([{"role": "assistant", "content": "hi"}], id="no user message"),
            pytest.param([{"role": "user", "content": "   "}], id="empty query"),
            pytest.param(
                [
                    {"role": "system", "content": h.MEMORY_MARKER + " earlier"},
                    {"role": "user", "content": "hi"},
                ],
                id="memory already injected",
            ),
        ],
    )
    def test_namespace_header_with_a_good_token_is_refused(self, signing, messages):
        """The header rule holds on the same requests as the bad-token rule, so
        whether a request gets a 401 does not depend on its message shape."""
        data = {
            "model": "memmachine-m",
            "messages": messages,
            "proxy_server_request": {
                "headers": {"x-openwebui-user-jwt": token(), "x-memmachine-org": "bob"}
            },
        }
        with pytest.raises(h.MemMachineRequestError) as caught:
            asyncio.run(h.MemMachineHook().async_pre_call_hook(None, None, data, "completion"))
        assert caught.value.status_code == 401

    def test_no_token_still_returns_early(self, signing):
        data = {
            "model": "memmachine-m",
            "messages": [{"role": "assistant", "content": "hi"}],
            "proxy_server_request": {"headers": {}},
        }
        out = asyncio.run(h.MemMachineHook().async_pre_call_hook(None, None, data, "completion"))
        assert out is data


class TestConfigurationFaultsAreNotTokenFaults:
    """A fault in the deployment must not be reported as a fault in the token."""

    def _import_with(self, **env):
        import subprocess, sys
        full = dict(os.environ, MEMMACHINE_BASE_URL="http://memmachine.unit.test", **env)
        if "FORWARD_USER_INFO_HEADER_JWT_SECRET" not in env:
            full.pop("FORWARD_USER_INFO_HEADER_JWT_SECRET", None)
        return subprocess.run([sys.executable, "-c", "import memmachine_hook"],
                              env=full, capture_output=True, text=True)

    def test_bad_leeway_does_not_touch_a_deployment_not_using_this(self):
        """With the secret unset, nothing in this feature may affect loading."""
        done = self._import_with(MEMMACHINE_FORWARDED_USER_JWT_LEEWAY="30s")
        assert done.returncode == 0, done.stderr[-400:]

    def test_bad_leeway_is_refused_when_the_feature_is_on(self):
        done = self._import_with(FORWARD_USER_INFO_HEADER_JWT_SECRET=SECRET,
                                 MEMMACHINE_FORWARDED_USER_JWT_LEEWAY="30s")
        assert done.returncode != 0
        assert "whole number of seconds" in done.stderr

    def test_a_non_pyjwt_failure_is_not_reported_as_a_bad_token(self, signing, monkeypatch):
        """A wrong argument or a shadowed package is a server error, not a 401.

        Reporting it as a bad token would turn a broken install into a 401 on
        every request, which looks like an auth problem and is not one.
        """
        class Shadowed:
            PyJWTError = jwt.PyJWTError

            @staticmethod
            def decode(*args, **kwargs):
                raise TypeError("decode() got an unexpected keyword argument")

        monkeypatch.setattr(h, "jwt", Shadowed)
        data, key = request({"x-openwebui-user-jwt": token()})
        with pytest.raises(TypeError):
            h._identity(data, key)


class TestStorePathLogsAnAuthFailure:
    def test_bad_token_is_logged_as_authentication(self, signing, monkeypatch):
        records = []
        monkeypatch.setattr(h, "_log_error", lambda **kw: records.append(kw))
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": "memmachine-m"},
                "proxy_server_request": {"headers": dict(FORGED)},
            },
            "messages": [{"role": "user", "content": "hi"}],
        }
        response = {"choices": [{"message": {"content": "hello"}}]}
        now = datetime.datetime.now()
        asyncio.run(h.MemMachineHook().async_log_success_event(kwargs, response, now, now))
        assert [r["category"] for r in records] == ["authentication"]
