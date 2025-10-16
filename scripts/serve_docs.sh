#!/bin/bash

# Script to serve the docs folder using Python's HTTP server
cd "$(dirname "$0")/../docs"
echo "Serving docs at http://localhost:8000"
python -m http.server 8000
