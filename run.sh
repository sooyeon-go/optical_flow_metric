CUDA_VISIBLE_DEVICES=7 python /data/project-vilab/sy/optical_flow_metric/score_video_motion.py \
  --video_dir /data/shared-vilab/datasets/OpenVid-1M/video \
  --weight_dir /data/project-vilab/sy/optical_flow_metric/weight \
  --max_videos 3 \
  --save_json /data/project-vilab/sy/test_spotting/optical_flow_metric/output/result_debug_3.json