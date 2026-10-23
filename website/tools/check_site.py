"""Validate static links, example-mask coverage, download integrity and scientific counts."""

from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json, hashlib, re, struct

P = Path(__file__).resolve().parents[1]
S = P / "site"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            assert a["id"] not in self.ids, ("Duplicate ID", a["id"])
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])


parsers = {}
for f in S.glob("*.html"):
    parser = Page()
    parser.feed(f.read_text())
    parsers[f] = parser
count = 0
for file, page in parsers.items():
    for ref in page.refs:
        url = urlsplit(ref)
        if url.scheme in ("https", "http", "mailto"):
            continue
        target = (file.parent / unquote(url.path)).resolve() if url.path else file
        assert target.is_relative_to(S.resolve()), ("Escaping path", ref)
        assert target.is_file(), ("Broken resource", file.name, ref)
        if url.fragment and target.suffix == ".html":
            assert url.fragment in parsers[target].ids, ("Broken anchor", ref)
        count += 1
for f in S.rglob("*.css"):
    for url in re.findall(r'url\([\'"]?([^\)\'\"]+)', f.read_text()):
        assert (f.parent / url).is_file(), ("Missing CSS asset", url)
raw = (S / "data.js").read_text()
data = json.loads(raw.removeprefix("window.KU_PCPE = ").strip().removesuffix(";"))
assert len(data["classes"]) == 9 and len(data["examples"]) == 18
assert sum(c["train"] for c in data["classes"].values()) == 4093
assert sum(c["test"] for c in data["classes"].values()) == 1327
assert data["stats"] == dict(
    images=4242,
    tasks=5420,
    masks=8074,
    classes=9,
    trainTasks=4093,
    testTasks=1327,
    testAnnotations=3981,
)
assert (
    data["stats"]["trainTasks"] + data["stats"]["testAnnotations"]
    == data["stats"]["masks"]
)
assert set(data["examples"]) == {
    c + "-" + kind for c in data["classes"] for kind in ("typical", "ambiguous")
}
for key, example in data["examples"].items():
    assert (S / example["image"]).is_file()
    assert set(example["overlays"]) == {
        "expert1",
        "expert2",
        "expert3",
        "strict",
        "aligned",
    }
    for mask in example["overlays"].values():
        content = (S / mask).read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", content[16:24]) == (
            example["width"],
            example["height"],
        ), ("Overlay dimensions", key, mask)
        assert content[25] == 6, ("Expected RGBA mask", mask)
    assert 0 <= example["pairDice"] <= 1
assert len(data["benchmarks"]["full"]) == 7 and len(data["benchmarks"]["matched"]) == 4
for rows in data["benchmarks"].values():
    for row in rows:
        assert 0 <= row["low"] <= row["dice"] <= row["high"] <= 1
# Check the printed headline scores as well as the complete content digests.
assert (
    round(
        next(
            r["dice"]
            for r in data["benchmarks"]["full"]
            if r["label"] == "Greedy Oracle@3"
        ),
        3,
    )
    == 0.672
)
assert (
    round(
        next(
            r["dice"]
            for r in data["benchmarks"]["matched"]
            if r["label"] == "Template@3"
        ),
        3,
    )
    == 0.402
)
assert (
    round(
        next(
            r["dice"]
            for r in data["benchmarks"]["matched"]
            if r["label"] == "Qwen3-VL-32B"
        ),
        3,
    )
    == 0.395
)
files = json.loads((P / "CONTENT_MANIFEST.json").read_text())
for name, digest in files["sha256"].items():
    assert hashlib.sha256((S / name).read_bytes()).hexdigest() == digest, (
        "Changed verified content",
        name,
    )
assets = json.loads((P / "ASSET_MANIFEST.json").read_text())["sha256"]
actual = {f.relative_to(S).as_posix() for f in (S / "assets").rglob("*") if f.is_file()}
assert actual == set(assets), "Unlisted or missing website asset"
for name, digest in assets.items():
    path = S / name
    assert (
        path.resolve().is_relative_to((S / "assets").resolve())
        and not path.is_symlink()
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
        "Changed website asset",
        name,
    )
assert not re.search(r"https?://", (S / "assets/fonts/fonts.css").read_text()), (
    "Fonts must be self-hosted"
)
assert '<script src="http' not in (S / "index.html").read_text()
assert not re.search(
    r"/home/|/share/|ghp_[A-Za-z0-9]+|-----BEGIN.*PRIVATE KEY",
    "\n".join(
        f.read_text() for f in S.glob("*") if f.suffix in (".html", ".css", ".js")
    ),
)
print(
    json.dumps(
        dict(
            status="PASS",
            html_pages=len(parsers),
            internal_links=count,
            examples=18,
            transparent_mask_overlays=90,
            classes=9,
            verified_downloads=3,
            benchmark_modes=2,
            runtime_assets_local=True,
        )
    )
)
