#!/usr/bin/env bash

# This script installs the required system dependencies for reportlab.

# Update the package list
apt-get update

# Install the dependencies needed by reportlab
apt-get install -y libffi-dev libjpeg-dev zlib1g-dev
