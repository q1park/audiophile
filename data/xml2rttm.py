# xml2rttm.py words/ES2008a.*.words.xml | sort -n -k 4 > rttm/ES2008a.rttm

import math
import sys
import xml.etree.ElementTree as ET

xml_trees = []

def print_rttm_line(start_time, end_time, speaker_num):
    print("SPEAKER meeting\t1\t{}\t{}\t<NA>\t<NA>\tSpeaker_{}\t<NA>".format(start_time, \
                                                                            end_time - start_time, speaker_num))

def convert_xml_to_rttm():
    xml_file = xml_files[xml_file_index]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_trees.append(root)

    start_time = end_time = None

    for element in root:
        if element.tag == 'w' and 'starttime' in element.attrib and 'endtime' in element.attrib:
            if start_time is None:
                start_time = float(element.attrib['starttime'])
            if end_time is None:
                end_time = float(element.attrib['starttime'])  # yes, 'starttime'
            if math.isclose(end_time, float(element.attrib['starttime']), abs_tol=0.01):
                # collapse the two
                end_time = float(element.attrib['endtime'])
            else:
                print_rttm_line(start_time, end_time, speaker_num=xml_file_index)
                start_time = float(element.attrib['starttime'])
                end_time = float(element.attrib['endtime'])
    if not ((start_time is None) or (end_time is None)):
        print_rttm_line(start_time, end_time, speaker_num=xml_file_index)

xml_files = sys.argv[1:]

for xml_file_index in range(len(xml_files)):
    convert_xml_to_rttm()