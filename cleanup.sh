#!/bin/bash

# Remove build artifacts
rm -rf dist
rm -rf build
rm -rf *.egg-info
rm -rf src/*.egg-info

# Remove macOS .DS_Store files
find . -name ".DS_Store" -delete
