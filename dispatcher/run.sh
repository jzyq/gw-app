#!/bin/bash

source /usr/local/Ascend/driver/bin/setenv.bash

# mkdir /dev/shm/dmp
# mkdir /home/HwHiAiUser/hdc_ppc
# nohup /var/dmp_daemon -I -M -U 8087 >&/dev/null &
# /var/slogd -d

python task_dispatcher.py
