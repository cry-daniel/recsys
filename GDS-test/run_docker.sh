docker run --gpus all -it \
    --name gr_inference --ipc host \
    --privileged \
    --ulimit memlock=-1 \
    -v /docker:/docker \
    -v /home/ruiyang.chen/Code/GR/recsys:/workspace/recsys \
    -v /usr/local/cuda-13:/usr/local/cuda-13 \
    -w /workspace/recsys \
   recsys-inference:latest