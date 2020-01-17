# run on cluster and it will open a window in local chrome

export XDG_RUNTIME_DIR=""
jupyter notebook --no-browser --port=8880 --ip=0.0.0.0
