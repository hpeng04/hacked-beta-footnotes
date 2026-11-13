#!/usr/bin/env bash
# PreToolUse hook: run a Codex code review before `git push`.
# Blocks the push once per HEAD commit with the review output; pushing the
# same HEAD again goes through.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // ""')
grep -Eq '(^|[^[:alnum:]_-])git([[:space:]]+-[^[:space:]]+([[:space:]]+[^-][^[:space:]]*)?)*[[:space:]]+push([[:space:]]|$)' <<<"$cmd" || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found; skipping pre-push review." >&2
  exit 0
fi

head=$(git rev-parse HEAD 2>/dev/null) || exit 0
marker="$(git rev-parse --git-dir)/codex-reviewed"
[ "$(cat "$marker" 2>/dev/null)" = "$head" ] && exit 0

# Review unpushed commits; fall back to the default branch for new branches.
base=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) ||
  base=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null) ||
  base=origin/main
git rev-parse --verify --quiet "$base" >/dev/null || exit 0
[ "$(git rev-parse "$base")" = "$head" ] && exit 0

if ! review=$(codex review --base "$base" 2>&1); then
  echo "codex review failed; skipping pre-push review:" >&2
  echo "$review" | tail -n 20 >&2
  exit 0
fi

echo "$head" >"$marker"
{
  echo "Codex review of $base..HEAD (push blocked once for this commit):"
  echo
  echo "$review"
  echo
  echo "Fix any real issues and commit, or push again to proceed as-is."
} >&2
exit 2
