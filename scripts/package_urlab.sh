#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT="$ROOT/unreal/OpenArmRenderer/OpenArmRenderer.uproject"
ARCHIVE="${1:-$ROOT/unreal/runtime}"

if [[ -z "${UE_ROOT:-}" || ! -x "$UE_ROOT/Engine/Build/BatchFiles/RunUAT.sh" ]]; then
    echo "UE_ROOT must point to Epic's precompiled Unreal Engine 5.7 directory" >&2
    exit 2
fi
if [[ ! -f "$ROOT/unreal/OpenArmRenderer/Content/OpenArmRender.umap" ]]; then
    echo "Persistent /Game/OpenArmRender level is missing; run scripts/import_urlab_asset.sh" >&2
    exit 2
fi

if [[ -e "$ARCHIVE" ]]; then
    echo "Archive destination already exists; choose a new path or remove it explicitly: $ARCHIVE" >&2
    exit 2
fi
"$UE_ROOT/Engine/Build/BatchFiles/RunUAT.sh" BuildCookRun \
    -project="$PROJECT" -noP4 -platform=Linux -clientconfig=Development \
    -build -cook -map=OpenArmRender -stage -pak -archive \
    -archivedirectory="$ARCHIVE" -utf8output

test -x "$ARCHIVE/Linux/OpenArmRenderer/Binaries/Linux/OpenArmRenderer"
echo "$ARCHIVE/Linux/OpenArmRenderer/Binaries/Linux/OpenArmRenderer"
