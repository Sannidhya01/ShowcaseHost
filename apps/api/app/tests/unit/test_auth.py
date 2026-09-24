from app.services.auth import pkce_challenge, token_hash


def test_session_tokens_are_one_way_hashed() -> None:
    raw = "session-secret"
    assert token_hash(raw) != raw
    assert token_hash(raw) == token_hash(raw)


def test_pkce_challenge_is_url_safe() -> None:
    challenge = pkce_challenge("verifier")
    assert "=" not in challenge
    assert "+" not in challenge
    assert "/" not in challenge
