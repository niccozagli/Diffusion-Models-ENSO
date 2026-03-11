#!/bin/bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is not installed. Install it first from https://pixi.sh/"
  exit 1
fi

echo "Installing project environment with pixi..."
pixi install

echo "Setup complete. Useful commands:"
echo "  pixi run test"
echo "  pixi run format"
echo "  pixi run typecheck"
echo "  pixi run notebook"
