import os
from config import cfg

# TODO: Make the next four functions flexible
def remove_previous_frames():
    cmd = 'rm ' + cfg.frames_folder + '/*'
    os.system(cmd)
    cmd = 'rm ' + cfg.annotated_frames_folder + '/*'
    os.system(cmd)
    
def video_to_frames():
    cmd = 'ffmpeg -y -i ' + cfg.input_path + ' ' + cfg.frames_folder + \
            '/%5d.ppm'
    os.system(cmd)

def remove_previous_results():
    cmd = 'rm ' + cfg.result_folder + '/*'
    os.system(cmd)

def frames_to_video():

    cmd = 'ffmpeg -y -start_number 1 -framerate 2 -i ' + \
        cfg.annotated_frames_folder + '/%5d.jpg -vcodec mpeg4 ' + \
        '-pix_fmt yuvj422p ' + cfg.result_folder + '/test.avi'
    os.system(cmd)
