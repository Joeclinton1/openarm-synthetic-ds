#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT="$ROOT/unreal/OpenArmRenderer/OpenArmRenderer.uproject"
PLUGIN="$ROOT/unreal/OpenArmRenderer/Plugins/UnrealRoboticsLab"
URLAB_REPO="https://github.com/URLab-Sim/UnrealRoboticsLab.git"
URLAB_COMMIT="567cbd907a570b820beb87fbddd69c356a6d86da"

if [[ -z "${UE_ROOT:-}" ]]; then
    echo "UE_ROOT must point to Epic's precompiled Unreal Engine 5.7 directory" >&2
    exit 2
fi
if [[ ! -x "$UE_ROOT/Engine/Binaries/Linux/UnrealEditor" ]]; then
    echo "UnrealEditor is missing below UE_ROOT=$UE_ROOT" >&2
    exit 2
fi
if [[ ! -x "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" ]]; then
    echo "Unreal 5.7 Build.sh is missing below UE_ROOT=$UE_ROOT" >&2
    exit 2
fi
if ! cmake --version | head -1 | grep -Eq '3\.(2[4-9]|[3-9][0-9])|[4-9]\.'; then
    echo "URLab requires CMake 3.24 or newer" >&2
    exit 2
fi

mkdir -p "$(dirname "$PLUGIN")"
if [[ ! -d "$PLUGIN/.git" ]]; then
    git clone --recurse-submodules "$URLAB_REPO" "$PLUGIN"
fi
git -C "$PLUGIN" fetch origin "$URLAB_COMMIT"
git -C "$PLUGIN" checkout --detach "$URLAB_COMMIT"
git -C "$PLUGIN" submodule update --init --recursive
test "$(git -C "$PLUGIN" rev-parse HEAD)" = "$URLAB_COMMIT"
PATCH="$ROOT/patches/urlab-multi-worker-ports.patch"
if git -C "$PLUGIN" apply --check "$PATCH"; then
    git -C "$PLUGIN" apply "$PATCH"
elif ! git -C "$PLUGIN" apply --reverse --check "$PATCH"; then
    echo "Pinned URLab no longer matches the reviewed multi-worker port patch" >&2
    exit 2
fi

"$PLUGIN/third_party/build_all.sh" --engine "$UE_ROOT"
"$PLUGIN/Scripts/build_and_test_linux.sh" --engine "$UE_ROOT" --project "$PROJECT"

cd "$ROOT"
uv sync --python 3.12 --extra urlab --extra dev
uv run openarm-retarget prepare-urlab-asset
uv run openarm-retarget urlab-doctor --plugin "$PLUGIN"
