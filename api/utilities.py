import os

# TODO: Make the next four functions flexible
def remove_previous_frames():
    os.system('rm uploads/annotated-frames/*')
    os.system('rm uploads/frames/*')
    
def video_to_frames():
    os.system('ffmpeg -i uploads/test.avi uploads/frames/%5d.ppm')

def remove_previous_results():
    os.system('rm uploads/result/*')

def frames_to_video():
    os.system('ffmpeg -start_number 1 -framerate 2 -i ' + \
                'uploads/annotated-frames/%5d.jpg -vcodec mpeg4 ' + \
                '-pix_fmt yuvj422p uploads/result/test.avi')
