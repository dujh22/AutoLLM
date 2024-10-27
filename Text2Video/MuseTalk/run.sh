#CUDA_VISIBLE_DEVICES=1 /root/zhuxiaoxu/abc/miniconda3/envs/gs/bin/python -m scripts.inference --inference_config configs/inference/test.yaml
#CUDA_VISIBLE_DEVICES=1 /root/zhuxiaoxu/abc/miniconda3/envs/gs/bin/python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --batch_size 4 
CUDA_VISIBLE_DEVICES=1 /root/zhuxiaoxu/abc/miniconda3/envs/gs/bin/python -m scripts.realtime_inference_streaming  --inference_config configs/inference/realtime_streaming.yaml --batch_size 4
#CUDA_VISIBLE_DEVICES=1 /root/zhuxiaoxu/abc/miniconda3/envs/gs/bin/python -m scripts.realtime_inference_streaming  --audio_dir data/audio_streaming --batch_size 4
