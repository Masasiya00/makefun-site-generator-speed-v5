# makefun-site-generator-speed-v5

[![make.fun](https://make.fun/badge/bounty/site-generator-speed-01-run-20261122)](https://make.fun/bounty/site-generator-speed-01-run-20261122)
[![make.fun](https://make.fun/badge/owner/Masasiya00/makefun-site-generator-speed-v5)](https://make.fun/bounty/site-generator-speed-01-run-20261122)
[![make.fun](https://make.fun/badge/repo/Masasiya00/makefun-site-generator-speed-v5)](https://make.fun/bounty/site-generator-speed-01-run-20261122)

A fast static-site generator for the [make.fun bounty "Build a static site faster"](https://make.fun/bounty/site-generator-speed-01-run-20261122) (`site-generator-speed-01-run-20261122`).

## What it does

Given a site description — templates, Markdown pages and static files — it builds
the whole site: HTML pages, index and tag pages with pagination, a tag index,
and a `build-report.txt`, byte-for-byte identical to the reference `makefun-site`
generator. It implements the project's own Markdown dialect (front matter,
template language, slug rules) from scratch, in C.

## How it is fast

- **One native engine.** All parsing, rendering and slug work happens in a
  single C addon (`addon/engine.c`, N-API), called once per run. No
  per-page JavaScript round-trips, no intermediate DOM.
- **Schema-first input parsing.** The 2.9 MB `cases.json` is parsed directly
  into the structures the generator needs — no generic JSON tree is built.
- **Arena allocation.** Two bump arenas (a work arena and a separate arena
  for the parsed input model) replace per-string allocation; the output is
  written with a single `write()` per case.
- **Single-pass rendering.** Markdown and templates render straight into
  the output buffer; template tag lines that stand alone are stripped in a
  line pass before parsing.
- **PGO.** `addon/build.sh --pgo` rebuilds the addon with profile-guided
  optimization trained on the practice cases.

## Layout

- `main.ts` — thin driver: reads `cases.json`, calls `run()` per case,
  writes `answers.json`.
- `solve.js` — loads the native addon and exposes `run(input)` / `runAll`.
- `addon/engine.c` — the whole generator (JSON parse, front matter,
  Markdown, templates, slugs, collections, report).
- `addon/build.sh` — builds `addon/engine.node` (`gcc -O3 -march=x86-64-v2
  -flto`; add `--pgo` for the profile-guided build).
- `addon/slug_tables.h` — Unicode NFKD decomposition tables for slug
  generation, generated from Python's `unicodedata`.

## Verify

```sh
# practice cases (3/3 byte-identical vs the reference answers)
cd addon && ./build.sh
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=/tmp/out \
  node main.ts

# full local grading
cd <kit> && node cli/makefun.mjs grade-local --kit . \
  --bot <this repo> --driver process
```

## License

MIT — see `LICENSE`.
