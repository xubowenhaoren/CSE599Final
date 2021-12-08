# https://towardsdatascience.com/face-mask-detection-using-darknets-yolov3-84cde488e5a1
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm


def get_yolo_format_size(box, width, height):
    point_a, point_b, point_c, point_d = box
    w = abs(point_a - point_c)
    h = abs(point_b - point_d)
    x = (point_a + point_c) / 2
    y = (point_b + point_d) / 2
    return " ".join([str(x / width), str(y / height), str(w / width), str(h / height)])


def xml_to_yolo(input_path, output_path):
    progress_bar = tqdm(glob.glob(input_path + '/*.xml'))
    for xml_file in progress_bar:
        root = ET.parse(xml_file).getroot()
        res = []
        for member in root.findall("object"):
            size_arr = root.find("size")
            width, height = int(size_arr[0].text), int(size_arr[1].text)
            class_name = member[0].text

            box = []
            for i in range(4):
                box.append(float(member[5][i].text))
            yolo_size_str = get_yolo_format_size(box, width, height)
            class_map = "0" if class_name == "with_mask" else "1"
            class_map += " "
            res.append(class_map + yolo_size_str + "\n")
        output_file_name = root.find("filename").text.split('.')[0] + ".txt"
        progress_bar.set_description("Converting " + output_file_name)
        with open(output_path + "/" + output_file_name, "w") as output_file:
            for output_str in res:
                output_file.write(output_str)
    print("all complete!")


if __name__ == '__main__':
    in_path = "archive/annotations"
    out_path = "archive/yolo_annotations"
    xml_to_yolo(in_path, out_path)
