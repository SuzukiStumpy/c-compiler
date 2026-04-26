#!/bin/bash
cd "${WORKSPACE_DIR:-.}"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
bash
