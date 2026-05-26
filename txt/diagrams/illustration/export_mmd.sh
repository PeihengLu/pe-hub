#!/bin/bash
cd "$(dirname "$0")"

ENCODED=$(base64 -w 0 < database_er.mmd)
curl -sf "https://mermaid.ink/img/${ENCODED}?type=png" -o database_er.png \
  && echo "Exported database_er.png" \
  || echo "Error: failed to render diagram" >&2
