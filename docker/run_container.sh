docker container run \
  -dit \
  --rm \
  -p 8990:8888 \
  --name practice-transformer \
  --user root \
  --mount type=bind,source=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)/../,target=/home/jovyan/work \
  --mount type=bind,source=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)/../../DATA/practice-transformer,target=/home/jovyan/work/data \
  kmrachet/practice-transformer:latest