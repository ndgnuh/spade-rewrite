#!/bin/sh
docker-compose -f .docker/docker-compose.yml up -d
docker exec -it -e HOME=/tmp -u$(id -u $USER) $@ spade-dev bash
