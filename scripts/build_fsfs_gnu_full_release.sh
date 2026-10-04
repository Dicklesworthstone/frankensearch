#!/usr/bin/env bash
# Build the standard (semantic-loader) x86_64-unknown-linux-gnu fsfs release
# binary inside Ubuntu 24.04, so its glibc floor is 2.39 instead of the build
# host's glibc (2.43 for v1.10.0 and v1.12.1, which shut out Ubuntu 22.04/24.04
# and Debian 12; GH #62). The prebuilt ONNX Runtime archive itself needs glibc
# 2.38 (C23 strtol family), so 24.04 is the oldest Ubuntu LTS that can link it.
#
# Usage: scripts/build_fsfs_gnu_full_release.sh [--source DIR] [--fast-cmaes DIR] [--cache DIR]
#
#   --source DIR      clean checkout to build (default: this repository)
#   --fast-cmaes DIR  sibling path dependency of tools/optimize_params
#                     (default: <source>/../fast_cmaes)
#   --cache DIR       container cargo target + registry cache (default: /data/tmp/fsfs-gnu-release)
#
# Prints the built binary's path. Fails unless the binary needs no glibc symbol
# newer than 2.39 and reports its version inside a stock Ubuntu 24.04 image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "$SCRIPT_DIR/.." && pwd)"
FAST_CMAES=""
CACHE="/data/tmp/fsfs-gnu-release"
MAX_GLIBC="2.39"
BASE_IMAGE="ubuntu:24.04"
UA="OpenAI File Downloader, XaiImageApiFetch/1.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$(cd "$2" && pwd)"; shift 2 ;;
    --fast-cmaes) FAST_CMAES="$(cd "$2" && pwd)"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
FAST_CMAES="${FAST_CMAES:-$(cd "$SOURCE/.." && pwd)/fast_cmaes}"
[[ -f "$FAST_CMAES/Cargo.toml" ]] || { echo "fast_cmaes not found at $FAST_CMAES (use --fast-cmaes)" >&2; exit 2; }

toolchain="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SOURCE/rust-toolchain.toml")"
[[ -n "$toolchain" ]] || { echo "no channel in $SOURCE/rust-toolchain.toml" >&2; exit 2; }
image="fsfs-gnu-release-builder:${toolchain}"
mkdir -p "$CACHE/target" "$CACHE/registry" "$CACHE/image"

cat >"$CACHE/image/Dockerfile" <<DOCKERFILE
FROM $BASE_IMAGE
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      build-essential pkg-config perl curl git ca-certificates cmake clang xz-utils binutils \\
    && rm -rf /var/lib/apt/lists/*
RUN userdel -r ubuntu 2>/dev/null || true; groupadd -g $(id -g) builder && useradd -m -u $(id -u) -g $(id -g) builder
USER builder
ENV CARGO_HOME=/home/builder/.cargo RUSTUP_HOME=/home/builder/.rustup PATH=/home/builder/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN curl -fsSL -A "$UA" https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $toolchain
DOCKERFILE
docker build -q -t "$image" "$CACHE/image" >/dev/null

echo "building fsfs ($(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || echo unknown-revision)) in $BASE_IMAGE with $toolchain" >&2
container="fsfs-gnu-release-$$"
docker run --name "$container" --user "$(id -u):$(id -g)" \
  -v "$SOURCE:/src/frankensearch:ro" -v "$FAST_CMAES:/src/fast_cmaes:ro" \
  -v "$CACHE/target:/target" -v "$CACHE/registry:/home/builder/.cargo/registry" \
  -e CARGO_TARGET_DIR=/target -w /src/frankensearch \
  "$image" cargo build --locked --release -p frankensearch-fsfs >&2
docker rm "$container" >/dev/null

binary="$CACHE/target/release/fsfs"
floor="$(objdump -T "$binary" | grep -oE 'GLIBC_2\.[0-9]+' | sed 's/GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -n 1)"
if [[ "$(printf '%s\n%s\n' "$floor" "$MAX_GLIBC" | sort -t. -k1,1n -k2,2n | tail -n 1)" != "$MAX_GLIBC" ]]; then
  echo "glibc floor $floor exceeds $MAX_GLIBC" >&2
  exit 1
fi
container="fsfs-gnu-release-check-$$"
version="$(docker run --name "$container" --user "$(id -u):$(id -g)" -v "$binary:/usr/local/bin/fsfs:ro" "$BASE_IMAGE" fsfs version)"
docker rm "$container" >/dev/null
echo "glibc floor $floor; $BASE_IMAGE reports: $version" >&2
echo "$binary"
