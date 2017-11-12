import os
import time
import xml.etree.cElementTree as ET
from xml.dom import minidom

from config import api_config


def video_to_frames(videoname):
    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    out_path = os.path.join(dir_path, 'frames')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, 'frames'))
        os.makedirs(os.path.join(dir_path, 'annotated-frames'))
        os.makedirs(os.path.join(dir_path, 'result'))
        os.makedirs(os.path.join(dir_path, 'annotations'))

        video_path = os.path.join(api_config.upload_folder, videoname)

        cmd = 'ffmpeg -y -i ' + video_path + ' -start_number 0 ' + out_path + \
              '/%d.jpg'
        os.system(cmd)
    return out_path


def frames_to_video(videoname):

    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    annotated_frames_folder = os.path.join(dir_path, 'annotated-frames')
    out_path = os.path.join(dir_path, 'result', videoname)
    cmd = 'ffmpeg -y -start_number 0 -framerate 2 -i ' + \
        annotated_frames_folder + '/%d.jpg -vcodec mpeg4 ' + \
        '-pix_fmt yuvj422p ' + out_path
    os.system(cmd)


def xml_setup(im_name, shape):

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = os.path.basename(im_name)
    ET.SubElement(annotation, "folder").text = "frames"

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "sourceImage").text = "video_input"
    ET.SubElement(source, "sourceAnnotation").text = "aivsol"

    imagesize = ET.SubElement(annotation, "imagesize")
    ET.SubElement(imagesize, "nrows").text = str(shape[0])
    ET.SubElement(imagesize, "ncols").text = str(shape[1])

    return annotation


def xml_add_object(annotation, frame, class_name, class_id, bbox):

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "deleted").text = "0"
    ET.SubElement(obj, "verified").text = "0"
    ET.SubElement(obj, "occluded").text = "no"
    ET.SubElement(obj, "attributes")

    parts = ET.SubElement(obj, "parts")
    ET.SubElement(parts, "hasparts")
    ET.SubElement(parts, "ispartof")
    ET.SubElement(obj, "date").text = time.strftime("%d-%m-%Y") + \
                                      " " + time.strftime("%H:%M:%S")
    ET.SubElement(obj, "id").text = str(class_id)
    ET.SubElement(obj, "type").text = "bounding_box"

    polygon = ET.SubElement(obj, "polygon")
    ET.SubElement(polygon, "username").text = "aivsol"

    pt = ET.SubElement(polygon, "pt")
    ET.SubElement(pt, "x").text = str(int(round(bbox[0])))
    ET.SubElement(pt, "y").text = str(int(round(bbox[1])))

    pt = ET.SubElement(polygon, "pt")
    ET.SubElement(pt, "x").text = str(int(round(bbox[2])))
    ET.SubElement(pt, "y").text = str(int(round(bbox[1])))

    pt = ET.SubElement(polygon, "pt")
    ET.SubElement(pt, "x").text = str(int(round(bbox[2])))
    ET.SubElement(pt, "y").text = str(int(round(bbox[3])))

    pt = ET.SubElement(polygon, "pt")
    ET.SubElement(pt, "x").text = str(int(round(bbox[0])))
    ET.SubElement(pt, "y").text = str(int(round(bbox[3])))


def xml_write(videoname, im_name, annotation):

    dir_path = os.path.join(api_config.upload_folder, videoname.split(".")[0])
    xmlstr = minidom.parseString(
                ET.tostring(annotation)).toprettyxml(indent="    ")
    annotation_xml_folder = os.path.join(dir_path, 'annotations')
    xml_file_name = im_name.replace("png", "xml")
    xml_file_path = os.path.join(annotation_xml_folder,
                                 xml_file_name)
    print 'Writing: ', xml_file_path
    with open(xml_file_path, "w") as f:
        f.write(xmlstr)
