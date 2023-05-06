import os
src_dir = "src"
videos = os.listdir(src_dir)
for video in videos:
    os.system(f"ffmpeg -y -i {src_dir}/{video} -vf scale=w=iw/4:h=ih/4. {video.replace('.avi','.mp4')}")
