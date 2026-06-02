# Videos 50,001 ~ 150,000 (skip first 50k, process next 100k)
CUDA_VISIBLE_DEVICES=7 python /data/project-vilab/sy/optical_flow_metric/score_video_motion.py \
  --video_dir /data/shared-vilab/datasets/OpenVid-1M/video \
  --weight_dir /data/project-vilab/sy/optical_flow_metric/weight \
  --skip_videos 50000 \
  --max_videos 100000 \
  --save_json /data/project-vilab/sy/optical_flow_metric/output/result_50k_150k.json
