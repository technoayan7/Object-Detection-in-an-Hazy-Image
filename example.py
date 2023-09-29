import cv2
import image_dehazer

thres = 0.55  # Threshold to detect objects

def remove_haze_and_detect_objects(input_image_path):
    # Remove Haze
    haze_img = cv2.imread(input_image_path)
    haze_corrected_img, haze_map = image_dehazer.remove_haze(haze_img, showHazeTransmissionMap=False)

    # Display the hazy image
    hazy_resized = cv2.resize(haze_img, (600, 500))
    cv2.imshow('Hazy Image', hazy_resized)

    # Display the dehazed image
    dehazed_resized = cv2.resize(haze_corrected_img, (600, 500))
    cv2.imshow('Dehazed Image', dehazed_resized)

    # Object Detection
    img = dehazed_resized  # Use the dehazed image for object detection

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 8, box[1] - 10),
                        cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    input_image_path = 'Images/horse.jpg'
    remove_haze_and_detect_objects(input_image_path)
