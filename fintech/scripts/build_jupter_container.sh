#!/bin/bash
# Launch an experiment using the docker cpu image
# https://github.com/NTUT-SELab/scripts/tree/master/stable-baselines

cmd_line="cd /root/code/env/ && jupyter notebook --generate-config && printf \"c.NotebookApp.ip='*'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port =8888\nc.NotebookApp.allow_root = True\nc.NotebookApp.token = ''\" > /root/.jupyter/jupyter_notebook_config.py"

echo "Executing in the docker (cpu image):"
echo $cmd_line


docker run -it --name jupyter  --ipc=host -p 8888:8888\
 --mount src=${SB_PATH}/stable_baselines/,target=/root/code/stable_baselines/,type=bind \
 --mount src=$(pwd),target=/root/code/env/,type=bind \
  fintechjupyter:latest \
  bash -c "jupyter notebook --generate-config && printf \"c.NotebookApp.ip='*'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port =8888\nc.NotebookApp.allow_root = True\nc.NotebookApp.token = ''\" > /root/.jupyter/jupyter_notebook_config.py"