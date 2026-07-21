import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from agentic_project_service.services.billing_cloud.jwt_minter import mint_billing_jwt


@pytest.fixture
def fake_private_key():
    """Generate ES256 keypair for testing."""
    private = ec.generate_private_key(ec.SECP256R1())
    return private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


@pytest.fixture
def fake_org_id():
    return uuid.uuid4()


@pytest.fixture
def fake_project_id():
    return uuid.uuid4()


def test_mint_includes_required_claims(monkeypatch, fake_private_key, fake_org_id, fake_project_id):
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", fake_private_key.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-test-v1")

    token = mint_billing_jwt(org_id=fake_org_id, project_id=fake_project_id)
    decoded = jwt.decode(token, options={"verify_signature": False})

    assert decoded["aud"] == "billing-service"
    assert decoded["iss"] == "powabase-control-plane"
    assert decoded["org_id"] == str(fake_org_id)
    assert decoded["sub"] == f"project:{fake_project_id}"
    assert decoded["scope"] == "billing:charge"
    assert decoded["exp"] > time.time() + 250  # 5 min exp


def test_mint_signs_with_es256_and_correct_kid(
    monkeypatch, fake_private_key, fake_org_id, fake_project_id
):
    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", fake_private_key.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-mytest-v3")
    token = mint_billing_jwt(org_id=fake_org_id, project_id=fake_project_id)
    headers = jwt.get_unverified_header(token)
    assert headers["alg"] == "ES256"
    assert headers["kid"] == "proj-mytest-v3"


def test_mint_signature_verifies_with_public_key(monkeypatch, fake_org_id, fake_project_id):
    """End-to-end: PS mints with private key, billing-service-like verifies with public key."""
    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )

    monkeypatch.setenv("BILLING_JWT_PRIVATE_KEY_PEM", private_pem.decode())
    monkeypatch.setenv("BILLING_JWT_KID", "proj-verify-v1")

    token = mint_billing_jwt(org_id=fake_org_id, project_id=fake_project_id)
    # Verify signature using the public key (simulates what billing service does)
    claims = jwt.decode(
        token,
        public_pem,
        algorithms=["ES256"],
        audience="billing-service",
        issuer="powabase-control-plane",
        options={"require": ["exp", "iss", "aud", "sub", "org_id"]},
    )
    assert claims["org_id"] == str(fake_org_id)
