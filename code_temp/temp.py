import torch
stats = torch.load("/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/pipline/ssl/results/test/teacher/teacher_resnet50_2ch_none_e25/rep_3/channel_stats.pt", map_location="cpu")
print(stats) 