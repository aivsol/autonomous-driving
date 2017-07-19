import os
from config import api_config


def prepare_directories(videoname):
    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, 'frames'))
        os.makedirs(os.path.join(dir_path, 'annotated-frames'))
        os.makedirs(os.path.join(dir_path, 'result'))


def video_to_frames(videoname):

    video_path = os.path.join(api_config.upload_folder, videoname)
    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    out_path = os.path.join(dir_path, 'frames')
    cmd = 'ffmpeg -y -i ' + video_path + ' ' + out_path + \
          '/%5d.ppm'
    os.system(cmd)
    return out_path


def frames_to_video(videoname):

    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    annotated_frames_folder = os.path.join(dir_path, 'annotated-frames')
    out_path = os.path.join(dir_path, 'result', videoname)
    cmd = 'ffmpeg -y -start_number 1 -framerate 2 -i ' + \
        annotated_frames_folder + '/%5d.jpg -vcodec mpeg4 ' + \
        '-pix_fmt yuvj422p ' + out_path
    os.system(cmd)
