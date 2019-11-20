"""
Testing opencv library.
Hough transformation [Probabilistic Line Transform and Standard Hough Line Transform]
"""
import sys
import os
import math
import cv2 as cv
import numpy as np
from merge import HoughBundler
from time import time
from shapely.geometry import Polygon


def main(argv):

    start = time()
    default_file = 'park.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    if not os.path.isfile(filename):
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    # Loads image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    # orientation line for filtering
    horizontal_line = None

    # sobel preprocessing
    grad = pre_process_sobel(src=src)

    # canny pre processign
    dst = pre_process_image(src=src)

    # backup copy for probabilistic Hough Lines
    cdstP = np.copy(grad)

    # obtain infinite lines from Hough transformation
    lines = cv.HoughLines(grad, 1, np.pi / 180, 150, np.array([]), 0, 0)

    # obtain lines from Probabilistic Hough Transformation
    cv.imshow("Probabilistic Hough input", dst)
    linesP = cv.HoughLinesP(dst, rho=1, theta=np.pi / 180,
                            threshold=100, lines=np.array([]),
                            minLineLength=80, maxLineGap=10)

    linesP_sobel = cv.HoughLinesP(grad, rho=1, theta=np.pi / 180,
                            threshold=100, lines=np.array([]),
                            minLineLength=80, maxLineGap=10)


    print("Size lines P_HoughLines:" + str(len(linesP)))

    # Preprocessing of hough lines with custom class for line filtering
    a = HoughBundler()
    foo = a.process_lines(linesP)
    print("Size lines bundled P_HoughLines:" + str(len(foo)))

    # copy for display result below
    src_copy = src.copy()

    # display results for different pre-processing
    display_hough_lines(src, a.process_lines(linesP_sobel),"sobel image hough")
    display_hough_lines(src_copy, a.process_lines(linesP),"canny image hough")

    marked = draw_probabilist_lines(cdstP, foo, horizontal_line)

    # Prova 1 - disegno parking space (2 posti attualmente)
    # Pre-processing per ordinare le coordinate
    print("dimenstion : {}".format(cdstP.shape))

    polylines_no_strech = algo_splitting_parking_lines(marked, cdstP)

    print("durata = " +str(time()-start))

    """ YOLO V3 """
    yolo_img, cars = get_yolooo(filename)
    
    yolo_img = draw_parking_spot(yolo_img, src.shape, polylines_no_strech,cars)
    
    # show final image
    cv.imshow("final image",yolo_img)
    cv.imwrite(filename.split('.')[0]+"_yolo.jpg",yolo_img)
    cv.waitKey()
    return 0


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    {\vec  r}\cdot {\vec  n}_{0}-d=0.\,"""
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def find_first_horizontal_line(lines):
    """ DEPRECATED """
    # search for horizontal lines
    for rho, theta in lines[0]:
        if theta < 2:   # if is horizontal line
            horizontal_line = [(rho, theta)]
            print("## HORIZONTAL LINE ##")
            print("theta: {}".format(theta))
            a = np.cos(theta)
            b = np.sin(theta)
            print("cos: {} and sin:{}".format(a, b))
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            pt1 = (x1, y1)
            pt2 = (x2,y2)

            # draw line if uncommented
            # cv.line(dst, pt1, pt2, (0,0,0), 10, cv.LINE_AA)

            print("(x1,y1): {} - (x2,y2): {}\n".format(pt1,pt2))
            # if found, break the loop
            return pt1,pt2


def display_hough_lines(img, lines, name_image):
    """ Display lines above image

    :param img : 3 channels image
    :param name_image: 3 channels image
    :param lines: [x,y,w,z] from the probabilisti Hough Lines
    :return: void method to draw some colored point in lines.
    """
    if lines is not None:
        line = []
        for l in lines:
            line.append(l[0][0])
            line.append(l[0][1])
            line.append(l[1][0])
            line.append(l[1][1])

            cv.line(img, (l[0][0], l[0][1]), (l[1][0], l[1][1]), (10, 10, 255), 2, cv.LINE_AA)
        cv.imshow(name_image, img)


def orientation(line):
    """ Deprecated """
    orientation = None
    x1, y1, x2, y2 = line

    if abs(x1 - x2) > abs(y1 - y2):
        orientation = 'horizontal'
    else:
        orientation = 'vertical'

    return orientation


def pre_process_sobel(src):
    """ SOBEL
    """
    # kernel format
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    kernel_9x9 = np.ones((9, 9), np.uint8)

    # canny parameters
    canny_low = 160
    canny_high = 230
    scale = 1
    delta = 0
    ddepth = cv.CV_8U

    src = cv.GaussianBlur(src, (3, 3), 0)

    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.3, abs_grad_y, 0.3, 0)
    grad = cv.Canny(grad, canny_low, canny_high, None, 3)
    grad = cv.morphologyEx(grad, cv.MORPH_CLOSE, kernel_9x9)
    cv.imshow("Sobel W+V", grad)
    return grad
    """END SOBEL """


def pre_process_image(src):
    # kernel format
    kernel_5x5 = np.ones((5,5),np.float32)/25
    kernel_9x9 = np.ones((9,9),np.uint8)

    # canny parameters
    canny_low = 160
    canny_high = 230
    blur = cv.filter2D(src, -1, kernel_5x5)
    dst = cv.Canny(blur, canny_low, canny_high, None, 3)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel_9x9)
    return dst


def draw_parking_spot(yolo_img, shape, polylines_no_strech,list_car):

    print("scr image shape {}".format(shape))
    (w_src, h_src) = shape

    print("scr image shape {}".format(yolo_img.shape[:2]))
    w_yolo, h_yolo = yolo_img.shape[:2] # yolo image has channels

    strech_w, strech_h = w_yolo/w_src, h_yolo/h_src
    print("strech rateo is {},{}".format(strech_w, strech_h))

    for i, pts in enumerate(polylines_no_strech):
        # choose a nice color for color parking spot
        free_park_color = (0,255,0)
        occupied_park_color = (0, 0, 255)  # bgr
        occupied_park_area = 4000  # empirical

        # print(pts)
        a0 = pts[0].take(0) # top left
        a0 *= strech_h
        
        a1 = pts[0].take(1)
        a1 *= strech_w

        a2 = pts[1].take(0) # top right
        a2 *= strech_h

        a3 = pts[1].take(1)
        a3 *= strech_w
        
        a4 = pts[2].take(0) # bot right
        a4 *= strech_h
        
        a5 = pts[2].take(1)
        a5 *= strech_w
        
        a6 = pts[3].take(0) # bot left
        a6 *= strech_h
        
        a7 = pts[3].take(1)
        a7 *= strech_w
       
        area_parking = abs(a0-a6)*abs(a1-a7)
        # print("car area: {}".format(list_car[0]))
        # print("parking area: {}".format(area_parking))

        if len(list_car)>0:
            area_car = list_car[0][2]*list_car[0][3]
            if area_parking/ area_car >= 2:
                pts_new1 = np.array(([a0,a1],[a2,a3],[(a2+a4)/2,(a3+a5)/2],[(a6+a0)/2,(a7+a1)/2]), np.int32)
                pts_new2 = np.array(([(a0+a6)/2,(a1+a7)/2],[(a2+a4)/2,(a3+a5)/2],[a4,a5],[a6,a7]), np.int32)

                cv.polylines(yolo_img,[pts_new1],True,free_park_color)
                if is_car_present_in_spot(pts_new1, list_car) and is_car_present_area(pts_new1,list_car)>occupied_park_area:
                    cv.fillPoly(yolo_img, [pts_new1], occupied_park_color)

                cv.polylines(yolo_img,[pts_new2],True,free_park_color)
                if is_car_present_in_spot(pts_new2, list_car) and is_car_present_area(pts_new2,list_car)>occupied_park_area:
                    cv.fillPoly(yolo_img, [pts_new2], occupied_park_color)
            else:
                print("car are too big for the image. BUt got a lines detection from hough ")
                pts_new = np.array(([a0, a1], [a2, a3], [a4, a5], [a6, a7]), np.int32)
                cv.fillPoly(yolo_img, [pts_new], free_park_color)
                cv.polylines(yolo_img, [pts_new], True, free_park_color)

        else:
            print("no car but display the parking spot from lines detection")
            pts_new = np.array(([a0,a1],[a2,a3],[a4,a5],[a6,a7]), np.int32)
            cv.fillPoly(yolo_img, [pts_new], free_park_color)
            cv.polylines(yolo_img,[pts_new],True,free_park_color)

    return yolo_img


def draw_probabilist_lines(img, lines, horizontal_line):
    # Filtering array declaration
    marked = []
    max_degree = []
    min_distance = 20
    a = HoughBundler()

    foo = lines
    cdstP = img
    if foo is not None:
        for i in range(0, len(foo)):
            l = foo[i]
            '''
            Documentazione provvisoria
            (xStart,yStart,xEnd,yEnd) hough probabilist result for a single line   
            '''
            line = []
            line.append(l[0][0])
            line.append(l[0][1])
            line.append(l[1][0])
            line.append(l[1][1])
            line = sort_coords(line)

            # Descriptor for line (values) and orientation
            print(line)
            print("\torientation theta: {}".format(a.get_orientation(line)))

            is_alone = True
            if marked == []:
                cv.line(cdstP, (l[0][0], l[0][1]), (l[1][0], l[1][1]), (10, 10, 255), 2, cv.LINE_AA)
                cv.circle(cdstP, (line[0], line[1]), 5, (255, 0, 0), -1)
                cv.circle(cdstP, (line[2], line[3]), 5, (0, 0, 255), -1)
                max_degree = a.get_orientation(line)
            else:
                for j in marked:
                    # delete overlapping lines
                    if a.get_distance(j, line) < min_distance and a.get_orientation(line) < max_degree + 20:
                        is_alone = False
                        break

            # drawing parking spots
            if is_alone and line[0] != 0:
                cv.line(cdstP, (l[0][0], l[0][1]), (l[1][0], l[1][1]), (10, 10, 255), 2, cv.LINE_AA)
                marked.append(line)
                # Draw starting and ending point of Hough line
                cv.circle(cdstP, (line[0], line[1]), 5, (255, 0, 0), -1)
                cv.circle(cdstP, (line[2], line[3]), 5, (0, 0, 255), -1)
            if horizontal_line:
                cv.line(cdstP, (l[0][0], l[0][1]), (l[1][0], l[1][1]), (10, 10, 255), 2, cv.LINE_AA)
    return marked


def algo_splitting_parking_lines(marked, img):
    polylines_no_strech = []
    cdstP = img

    for line in enumerate(marked):
        print("line "+str(line[0])+": "+ str(marked[line[0]]))

    for i in range(len(marked) - 1):
        f_top, f_left, f_bot, f_right = marked[i]
        s_top, s_left, s_bot, s_right = marked[i + 1]
        print("total area of parking: {}".format(get_area(marked[i], marked[i + 1])))
        pts = np.array(([f_top, f_left], [s_top, s_left], [s_bot, s_right], [f_bot, f_right]), np.int32)
        pts = pts.reshape((-1, 1, 2))

        if i % 2 == 0:
            color = (255, 0, 255)
        else:
            color = (0, 255, 0)

        cv.polylines(cdstP, [pts], True, (0, 255, 255))
        cv.fillPoly(cdstP, [pts], color)
        polylines_no_strech.append(pts)

        # put a measure of the car (car area) to split the parking spot
        rateo_img_car = 5
        if (cdstP.shape[0] * cdstP.shape[1]) / get_area(marked[i], marked[i + 1]) < rateo_img_car:
            if (i % 2) != 0:
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            pts = np.array([[f_top, f_left], [s_top, s_left], [(s_top + s_bot) / 2, (s_left + s_right) / 2],
                            [(f_top + f_bot) / 2, (f_left + f_right) / 2]], np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv.polylines(cdstP, [pts], True, color)
            cv.fillPoly(cdstP, [pts], color)
    return polylines_no_strech


def is_car_present_in_spot(pts, cars):
    """ check if need to fill the parking spot with color.

    :param pts: parking spot, edge points of the parking spot
    :param cars: list of all cars
    :return: Bool
    """
    a,b,c,d = pts
    polygon = Polygon([(a.take(0), a.take(1)), (b.take(0), b.take(1)), (c.take(0), c.take(1)), (d.take(0), d.take(1))])

    for i in cars:
        x,y,h,w = i
        other_polygon = Polygon([(x,y), (x,y+h), (x+w,y+h), (x+w,y)])
        intersection = polygon.intersects(other_polygon)
        if intersection:
            return True


def is_car_present_area(pts, cars):
    """ check if need to fill the parking spot with color.

    :param pts: parking spot, edge points of the parking spot
    :param cars: list of all cars
    :return: Bool
    """
    a,b,c,d = pts
    polygon = Polygon([(a.take(0), a.take(1)), (b.take(0), b.take(1)), (c.take(0), c.take(1)), (d.take(0), d.take(1))])
    print(polygon)

    area = 0
    for i in cars:
        x,y,w,h = i
        other_polygon = Polygon([(x,y), (x,y+w), (x+h,y+w), (x+h,y)])
        area_new = polygon.intersection(other_polygon).area
        if area_new > area :
            area = area_new

        print("area {}".format(area))
    return area


def get_area(line,line2):
    base = math.sqrt(abs(pow(line[0],2)-pow(line[2],2)) + abs(pow(line[1],2)-pow(line[3],2)))
    alt = math.sqrt(abs(pow(line[0],2)-pow(line[2],2)) + abs(pow(line[0],2)+pow(line[2],2)))

    return int((base*alt)/2)


def sort_coords(line):
    """ order set point of the line """
    a, b, c, d = line

    if b > c:
        a, b, c, d = c, d, a, b

    return [a,b,c,d]


def get_yolooo(filename):
    # Load Yolo
    print("\n\n  ######## YOLO v3 DETECTION ########")
    """    Needs a 3-channels image with dim:=(416,416)   """

    net = cv.dnn.readNet("darknet/yolov3.weights", "darknet/yolov3.cfg")

    classes = []

    with open("darknet/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image from the start of the script    
    src = cv.imread(cv.samples.findFile(filename))#, cv.IMREAD_GRAYSCALE)
    # resize image
    dim = (416, 416)
    img = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    # show input image for yolov3
    cv.imshow("YoloV3 image", img)

    height, width,channels = img.shape

    font = cv.FONT_HERSHEY_PLAIN
    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    area = 0
    cars = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car":
                cars.append([x, y, w, h])
            color = colors[i]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv.imshow("Image", img)
    if area > 0:
        return img, cars
    return img, cars


def draw_reference_rect(img):
    """ To draw a top left rect at 0,0 coords (testing reference XY)"""
    # creating a rectangle
    cv.rectangle(img, (0, 0), (50, 50), (0, 20, 200), 3)
    cv.imshow('rectangle reference image', img)


if __name__ == "__main__":
    main(sys.argv[1:])
