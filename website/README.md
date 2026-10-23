# KU-PCPE project page

The project page uses plain HTML, CSS and JavaScript, with no installation or build step. It includes the paper, supplement, examples for all nine classes, annotation comparisons and full-bank/matched-12 result charts.

## Preview and check

From the repository root:

```sh
python3 website/tools/check_site.py
node --check website/site/app.js
python3 -m http.server 4174 --bind 127.0.0.1 --directory website/site
```

Open `http://127.0.0.1:4174/`. Relative resource paths support GitHub Pages under `/ku-pcpe/`. Fonts, images and scripts are served locally; the page uses no analytics, accounts or cookies.

## Content and provenance

- `site/index.html`: paper identity, scientific text, one dataset download and two separately labeled reproduction assets.
- `site/style.css` and `site/app.js`: responsive layout, annotation comparisons, mask slider, benchmark switching and citation copy.
- `site/data.js`: counts, 18 paper examples, nine classes and full/matched-12 results.
- `site/credits.html` and `NOTICE.md`: research, image and font attribution.
- `ASSET_PROVENANCE.json`: selected tasks and research-input hashes.
- `CONTENT_MANIFEST.json`: numerical data and PDF/citation hashes.
- `ASSET_MANIFEST.json`: hashes for every static asset, including 18 resized illustrations, nine thumbnails and 90 overlays.

Update the corresponding manifests when intentionally changing verified content. The checker rejects missing or changed assets rather than regenerating reference hashes automatically.

## GitHub Pages deployment

The manual **Prepare project page** workflow packages `website/site/` with `index.html` at the artifact root. It validates links, assets and JavaScript first. There is no deployment on push. Running it with **deploy unchecked** produces an artifact for inspection. The deployment job requires a public repository and the `main` branch; its condition prevents deployment from private repositories.

To deploy from a public repository:

1. In Settings → Pages, select **GitHub Actions** as the source.
2. On `main`, run **Prepare project page** with **deploy** checked.
3. Use the URL returned by the deployment job. The configured project path is `https://adadras.github.io/ku-pcpe/`.

The original page code is MIT licensed. PDFs, annotations, illustrative photographs and fonts retain the terms described in the notices.
