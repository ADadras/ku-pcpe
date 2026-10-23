"""The website exception must not admit new or altered photo assets."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[1] / "scripts/verify_release.py"
)
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


@pytest.fixture
def website(tmp_path):
    asset = tmp_path / "website/site/assets/examples/paper.webp"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"reviewed illustration")
    (tmp_path / "website/ASSET_MANIFEST.json").write_text(
        json.dumps(
            {
                "sha256": {
                    "assets/examples/paper.webp": hashlib.sha256(
                        asset.read_bytes()
                    ).hexdigest()
                }
            }
        )
    )
    return tmp_path, asset


def test_website_rejects_an_additional_photo(website):
    root, asset = website
    assert RELEASE.verify_website_assets(root) == {asset}
    asset.with_name("unreviewed.webp").write_bytes(b"additional photograph")
    with pytest.raises(ValueError, match="Unlisted or missing"):
        RELEASE.verify_website_assets(root)


def test_website_rejects_an_altered_illustration(website):
    root, asset = website
    asset.write_bytes(b"substituted photograph")
    with pytest.raises(ValueError, match="Unapproved"):
        RELEASE.verify_website_assets(root)
