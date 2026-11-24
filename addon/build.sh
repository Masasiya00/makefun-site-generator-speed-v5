#!/bin/sh
# Build addon/engine.node (N-API, ABI-stable across Node versions).
# Usage: ./build.sh [--pgo]   (with --pgo: profile-generate run + profile-use rebuild)
set -e
cd "$(dirname "$0")"
NODE_INC=/home/hatch/node22/include/node
if [ ! -d "$NODE_INC" ]; then NODE_INC=/usr/include/node; fi
CC=${CC:-gcc}
CFLAGS="-O3 -march=x86-64-v2 -flto -fPIC -shared -DNODE_GYP_MODULE_NAME=engine -I$NODE_INC -Wall -Wno-unused-function -Wno-unused-variable"
if [ "$1" = "--pgo" ]; then
  echo "== PGO generate =="
  $CC $CFLAGS -fprofile-generate -o engine.node engine.c
  echo "== PGO training run (practice cases) =="
  cd ..
  MAKEFUN_INPUT_DIR=/home/hatch/workspace/makefun/site-generator-speed-01-kit-dl/site-generator-speed-01-run-20261122-kit/practice/input \
  MAKEFUN_OUTPUT_DIR=/tmp/pgo-out /home/hatch/node22/bin/node main.ts || true
  mkdir -p /tmp/pgo-out
  cd addon
  echo "== PGO use =="
  $CC $CFLAGS -fprofile-use -fprofile-correction -o engine.node engine.c
  rm -f *.gcda
else
  $CC $CFLAGS -o engine.node engine.c
fi
ls -lh engine.node
