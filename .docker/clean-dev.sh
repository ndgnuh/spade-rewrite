#!/bin/sh
docker-compose -f .docker/docker-compose.yml stop
docker rm -f spade-dev
