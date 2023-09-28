#!/bin/bash

cd $(dirname $0)
################################################################################
export PROJECT=$(whoami)
CONTAINER="${PROJECT}_tetris_1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"


# Run the Docker container in the background.
# Any changes made to './docker-compose.yml' will recreate and overwrite the container.

EXIST_CONTAINER=$(docker ps |grep $CONTAINER)
if [ -z "$EXIST_CONTAINER" ]; then
  export COMPOSE_FILE=./docker-compose.yml
  if [ ! -z "$(which nvidia-smi)" ]; then
    export COMPOSE_FILE=${COMPOSE_FILE}:./docker-compose-gpu.yml
  fi
  docker-compose -p ${PROJECT} up -d
else
  echo "${CONTAINER}はすでに存在します"
fi

################################################################################
docker exec -i -t ${CONTAINER} bash

