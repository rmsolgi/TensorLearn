#!/bin/bash

# Remove build artifacts
rm -rf dist
rm -rf build
rm -rf *.egg-info

# Remove macOS .DS_Store files
find . -name ".DS_Store" -delete
