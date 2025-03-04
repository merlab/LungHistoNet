import os
import csv

observer1_dir = "labels_O1/Healthy/"
observer2_dir = "labels_O2/Healthy/"
intersection_dir = "Intersection_Labels/Healthy"
union_dir = "Union_Labels/Healthy"

# os.makedirs(intersection_dir, exist_ok=True)
# os.makedirs(union_dir, exist_ok=True)

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def get_prefix(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])  
    return file_name

def to_yolo_format(box, class_id=0):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [class_id, x_center, y_center, width, height]

observer1_files = os.listdir(observer1_dir)
observer2_files = os.listdir(observer2_dir)


observer1_by_prefix = {get_prefix(f): f for f in observer1_files}
observer2_by_prefix = {get_prefix(f): f for f in observer2_files}


common_prefixes = set(observer1_by_prefix.keys()).intersection(observer2_by_prefix.keys())
print(f"Found {len(common_prefixes)} common prefixes between observers.")


for prefix in common_prefixes:
    observer1_file = observer1_by_prefix[prefix]
    observer2_file = observer2_by_prefix[prefix]
    observer1_path = os.path.join(observer1_dir, observer1_file)
    observer2_path = os.path.join(observer2_dir, observer2_file)


    observer1_boxes = []
    with open(observer1_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)
            observer1_boxes.append([x1, y1, x2, y2])


    observer2_boxes = []
    with open(observer2_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)
            observer2_boxes.append([x1, y1, x2, y2])


    intersection_boxes = []
    matched_o2_boxes = [False] * len(observer2_boxes)

    for o1_box in observer1_boxes:
        best_iou = 0.1  
        best_match = None

        for i, o2_box in enumerate(observer2_boxes):
            if not matched_o2_boxes[i]:
                iou = calculate_iou(o1_box, o2_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = o2_box
                    matched_idx = i

        if best_match is not None:

            avg_box = [
                (o1_box[0] + best_match[0]) / 2,
                (o1_box[1] + best_match[1]) / 2,
                (o1_box[2] + best_match[2]) / 2,
                (o1_box[3] + best_match[3]) / 2
            ]
            intersection_boxes.append(avg_box)
            matched_o2_boxes[matched_idx] = True

    union_boxes = observer1_boxes.copy()
    for i, o2_box in enumerate(observer2_boxes):
        if not matched_o2_boxes[i]:  
            union_boxes.append(o2_box)

    intersection_path = os.path.join(intersection_dir, f"{prefix}.txt")
    with open(intersection_path, "w") as f:
        for box in intersection_boxes:
            yolo_box = to_yolo_format(box)
            f.write(f"{yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {yolo_box[4]}\n")

    union_path = os.path.join(union_dir, f"{prefix}.txt")
    with open(union_path, "w") as f:
        for box in union_boxes:
            yolo_box = to_yolo_format(box)
            f.write(f"{yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {yolo_box[4]}\n")

    print(f"Processed {prefix}: Intersection={len(intersection_boxes)} boxes, Union={len(union_boxes)} boxes")

print(f"Intersection labels saved to {intersection_dir}")
print(f"Union labels saved to {union_dir}")