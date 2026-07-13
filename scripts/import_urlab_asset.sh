#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT="$ROOT/unreal/OpenArmRenderer/OpenArmRenderer.uproject"

if [[ -z "${UE_ROOT:-}" || ! -x "$UE_ROOT/Engine/Binaries/Linux/UnrealEditor" ]]; then
    echo "UE_ROOT must point to Epic's precompiled Unreal Engine 5.7 directory" >&2
    exit 2
fi

cd "$ROOT"
uv run openarm-retarget prepare-urlab-asset
"$UE_ROOT/Engine/Binaries/Linux/UnrealEditor" "$PROJECT" -Unattended -NoSplash -stdout &
EDITOR_PID=$!
trap 'kill "$EDITOR_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if uv run python -c 'from urlab_client import URLabClient; c=URLabClient(); c.discover(); c.close()' \
        >/dev/null 2>&1; then
        uv run openarm-retarget import-urlab-asset
        exit 0
    fi
    sleep 1
done
echo "URLab editor bridge did not become ready within 120 seconds" >&2
exit 1
