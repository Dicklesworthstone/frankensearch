#!/usr/bin/env bats

# Regression tests for GH#36: the default Linux route silently fell back to a
# 20+ minute cargo source build because the full artifact 404s, and --lite
# always source-built instead of downloading the published fsfs-lite-* tarball.
#
# These exercise the CANONICAL installer (repo-root install.sh, the file the
# README curls), not the legacy scripts/install.sh, through its non-networked
# FSFS_INSTALL_CONTRACT_TEST seam.

load "helpers/common.bash"

ROOT_INSTALLER=""

setup() {
  ROOT_INSTALLER="$(repo_root)/install.sh"
}

contract() {
  FSFS_INSTALL_CONTRACT_TEST=1 bash "$ROOT_INSTALLER" "$@"
}

@test "artifact-name: --lite resolves the published fsfs-lite-* archive" {
  run contract artifact-name 1 1.6.0 x86_64-unknown-linux-musl tar.xz
  [ "$status" -eq 0 ]
  [ "$output" = "fsfs-lite-1.6.0-x86_64-unknown-linux-musl.tar.xz" ]
}

@test "artifact-name: default resolves the full archive name" {
  run contract artifact-name 0 1.6.0 x86_64-unknown-linux-musl tar.xz
  [ "$status" -eq 0 ]
  [ "$output" = "fsfs-1.6.0-x86_64-unknown-linux-musl.tar.xz" ]
}

@test "--lite no longer forces a source build (downloads the prebuilt artifact)" {
  run contract args --lite
  [ "$status" -eq 0 ]
  # from_source must be 0 so the artifact-download path runs; lite stays 1.
  [[ "$output" == *"from_source=0"* ]]
  [[ "$output" == *"lite=1"* ]]
}

@test "--lite --from-source still forces an explicit source build" {
  run contract args --lite --from-source
  [ "$status" -eq 0 ]
  [[ "$output" == *"from_source=1"* ]]
  [[ "$output" == *"lite=1"* ]]
}

@test "route: explicit lite falls back to source-lite (handled, not an error)" {
  run contract route 1 0 x86_64-unknown-linux-musl
  [ "$status" -eq 0 ]
  [ "$output" = "source-lite" ]
}

@test "route: default with no full artifact falls back to source-default" {
  run contract route 0 0 x86_64-unknown-linux-musl
  [ "$status" -eq 0 ]
  [ "$output" = "source-default" ]
}
