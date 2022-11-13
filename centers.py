import xml.etree.ElementTree as ET
import numpy as np
import sys

def find_xml_centers(file):

    file = file.replace(".tiff", "")
    t = file[-2:]
    file = "ground_truth/"+file.replace(t, ".xml")

    centers = np.empty((0, 3), float)

    tree = ET.parse(file)

    for detection in tree.iter('detection'):
        
        detection_t = int(detection.attrib.get('t'))

        if(detection_t == int(t)):
            detection_x = float(detection.attrib.get('x'))
            detection_y = float(detection.attrib.get('y'))
            detection_z = float(detection.attrib.get('z'))

            centers = np.append(centers, np.array([[detection_x, detection_y, detection_z]]), axis = 0)
            
    return centers


def find_marker_centers(file):
    filestream = open(file, "r")
    centers = np.empty((0, 3), float)

    for _ in zip(range(1), filestream): pass

    for line in filestream:
        currentline = line.split(",")
        centers = np.append(centers, np.array([[float(currentline[0]), float(currentline[1]), float(currentline[2])]]), axis = 0)
    
    return centers


def list_xml_centers(file_list):
    centers_list = []
    for file in file_list:
        centers_list.append(find_xml_centers(file))
    return centers_list
    

def list_marker_centers(file_list):
    centers_list = []
    for file in file_list:
        centers_list.append(find_marker_centers(file))
    return centers_list


def find_xml_centers_256(file):

    file = file.replace(".tiff", "")
    t = file[-2:]
    file = "ground_truth/"+file.replace(t, ".xml")

    centers = np.empty((0, 3), float)

    tree = ET.parse(file)

    for detection in tree.iter('detection'):
        
        detection_t = int(detection.attrib.get('t'))
        detection_x = float(detection.attrib.get('x'))
        detection_y = float(detection.attrib.get('y'))
        

        if(detection_t == int(t)):
            if(detection_x < 256 and detection_y < 256):
                detection_z = float(detection.attrib.get('z'))

                centers = np.append(centers, np.array([[detection_x, detection_y, detection_z]]), axis = 0)
            
    return centers

