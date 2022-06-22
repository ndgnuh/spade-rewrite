#!/bin/sh

if command -v nvidia-smi; then
	dockerfile='.docker/Dockerfile'
else
	dockerfile='.docker/cpu.Dockerfile'
fi

if ! [ -z $USE_CPU ]; then
	dockerfile='.docker/cpu.Dockerfile'
fi

echo Building with $dockerfile
VERSION=$(cat .docker/VERSION)
docker build -t "$VERSION" -f "$dockerfile" .
