#!/usr/bin/env bash
# Download the OpenStax books used for RAG (CC BY-NC-SA 4.0, free for non-commercial use).
# URLs come from the OpenStax CMS API: https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&slug=<slug>
set -euo pipefail
dest="${1:-data/books}"
mkdir -p "$dest"
while read -r name url; do
  [[ -f "$dest/$name" ]] && { echo "exists: $name"; continue; }
  echo "downloading: $name"
  curl -fL --retry 3 -o "$dest/$name" "$url"
done <<'LIST'
organizational-behavior.pdf https://assets.openstax.org/oscms-prodcms/media/documents/organizational-behavior_-_WEB.pdf
principles-of-management.pdf https://assets.openstax.org/oscms-prodcms/media/documents/principles-management_-_WEB.pdf
LIST
