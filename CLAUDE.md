# Weather Map Studio: notes for Claude

The owner is not a programmer. They ask for maps and changes in plain English through GitHub issues
and comments. Do the technical work, test it, and explain results in plain, short language.

## What this repo does
Makes detailed, broadcast-quality weather maps and animations for social media (X, Instagram,
TikTok) from free public data. Output goes to `out/`, which the workflows attach to each run as
an artifact the owner downloads.

## Layout
- `studio/__main__.py`: command line. `python -m studio <command> --dry-run` (dry run = never post).
  Commands: `storm-anim`, `upper-air`, `replay` (severe outbreak recap), `hazard`, `watch`,
  `anniversary`, `status`.
- `studio/products/`: one module per map type
  - `storm_anim.py`: storm-following hurricane animation (GOES IR from NOAA AWS 2018+,
    NCEI GridSat-B1 1983-2017; track from HURDAT2 or the live ATCF b-deck)
  - `upper_air.py`: GFS analysis, 500/250/850 mb maps with 1991-2020 anomalies
  - `severe.py`: radar (IEM NEXRAD composite) + warning polygons + storm reports loops/recaps
  - `hazards.py`: national hazard maps (zones from IEM, METAR obs, WPC fronts)
  - `tropical.py`: NHC active storms + triggers
- `studio/geo.py`: projections (Lambert, plate carree) and the bundled basemap
  (`studio/data/basemap.json`: coastlines, borders, states, lakes, cities)
- `studio/render.py`: branding bar and MP4 encoding (ffmpeg, H.264, yuv420p)
- `studio/captions.py`: post text written by Claude Haiku; `studio/x_client.py`: X posting
- `studio/config.py`: all settings and thresholds
- `tests/test_studio.py`: offline tests with synthetic data. Run `python -m pytest -q tests`.

## How to add a new map type
1. Add `studio/products/<name>.py` with a `build(out_path, ...) -> dict_of_facts` function.
2. Add a subcommand in `studio/__main__.py` that writes into `config.OUT_DIR`.
3. Add it to the `map` choices in `.github/workflows/make-map.yml`.
4. Add an offline test with synthetic data (no network in tests).
5. Render one real example into `out/` before finishing so the owner can see it in the artifacts.

## Visual style (keep everything consistent)
- Dark theme: background `#0e141e`/`#070b12`, white text, yellow `#ffd25a` for times/highlights.
- 16:9 frames: 1920x1080 for animations, 1600x1000 for still maps. Header with title + time,
  footer with data credit and `config.HANDLE or config.BRAND`.
- Always credit the data source on the image (NOAA/NWS/NHC/IEM).
- Category colors (Saffir-Simpson) and warning colors are defined in `storm_anim.py` / `severe.py`.
- Check every new image by opening it: no overlapping labels, nothing hidden under panels.

## Rules
- Use only free, public data (NOAA, NWS, NHC, IEM, NASA). No scraping of commercial sites.
- Never put links in post text (X charges about 13x more for posts with links).
- Never commit secrets. Keys live in GitHub Secrets.
- Keep `--dry-run` on unless the owner explicitly asks to post.
- Keep tests passing; run them before opening a pull request.
- In your final comment, say in 2-4 plain sentences what you changed and where to download results.
