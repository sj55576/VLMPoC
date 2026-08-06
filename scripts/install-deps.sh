#!/bin/bash
# Installs the project so cloud coding sessions can run make lint / make test.
# Local sessions are left alone: developers manage their own environment.
[ "$CLAUDE_CODE_REMOTE" = "true" ] || exit 0
python -m pip install -e ".[dev]" >/dev/null 2>&1 || echo "install-deps: pip install failed; run 'make install' manually" >&2
exit 0
