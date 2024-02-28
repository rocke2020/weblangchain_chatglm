# 
gpu=0
export MODEL_PATH=/mnt/nas1/models/THUDM/chatglm3-6b
file=chatglm3/openai_api_demo/openai_api.py
nohup python $file \
    > $file.log 2>&1 &