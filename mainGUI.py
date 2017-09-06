import tkinter as tk
import cv2
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import numpy as np
import math

root = tk.Tk()
root.geometry('900x700')
root.title('Calligraphy Segment tool')

img_Tk = None

img_current = None
img_previous = None

img_original = None

img_ = None
original_image = None
img_grayscale = None
img_threshold = None
img_denoise = None
img_current = None
rectangles = None

img_saved = None

scale = 0.0

canvas_width = 600
canvas_height = 600

var_threshold = tk.DoubleVar()
var_denoise = tk.DoubleVar()

btn_Open = tk.Button(root, text='Open', command=lambda: do_open(canvas))
btn_Binary = tk.Button(root, text='Binary', command=lambda: do_binary(canvas))
btn_Threshold_Save = tk.Button(root, text='Threshold Save', command=lambda: do_threshold_save())
btn_Segment = tk.Button(root, text='Segment', command=lambda: do_segment(canvas))
btn_Save = tk.Button(root, text='Save', command=lambda: do_save())


tk.Label(root, text='Threshold: ').grid(row=2, column=4, sticky=tk.W+tk.E)
scl_Threshold = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, variable=var_threshold,
                         resolution=5, command=lambda _: do_threshold(canvas))
scl_Threshold.set(127)

# tk.Label(root, text='Denoise: ').grid(row=3, column=4, sticky=tk.W+tk.E)
# scl_Denoise = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)

tk.Label(root, text='Separation: ').grid(row=4, column=4, sticky=tk.W+tk.E)
ent_Separation = tk.Entry(root)
ent_Separation.grid(row=4, column=5, sticky=tk.E+tk.W)


canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.grid(row=0, column=0, columnspan=4, rowspan=7, sticky=tk.W+tk.E, padx=5, pady=5)

scl_Threshold.grid(row=2, column=5, sticky=tk.W+tk.E)
# scl_Denoise.grid(row=3, column=5, sticky=tk.W+tk.E)

btn_Denoise = tk.Button(root, text='Denoise', command=lambda: do_denoise(canvas))
# btn_Denoise.grid(row=3, column=4, columnspan=2, sticky=tk.E+tk.W)

btn_Open.grid(row=0, column=4, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
btn_Binary.grid(row=1,column=4, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
btn_Threshold_Save.grid(row=3, column=4, columnspan=2, stick=tk.W+tk.E, padx=5, pady=5)
btn_Segment.grid(row=5, column=4, columnspan=2, stick=tk.W+tk.E, padx=5, pady=5)
btn_Save.grid(row=6, column=4, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)


def do_open(canvas):
    global img_current
    global img_previous
    global img_original

    global img_Tk

    global scale

    openfile = askopenfile(mode='r')
    if not openfile:
        print('File is None')
        return
    image_file = openfile.name
    if 'jpg' in image_file or 'jpeg' in image_file or 'png' in image_file:
        print(image_file)
        # image
        image = Image.open(image_file, mode='r')
        img_original = image
        w, h = image.size
        # resize
        if w > canvas_width or h > canvas_height:
            scale = canvas_width / max([w, h])
            w = int(w * scale)
            h = int(h * scale)
            image = image.resize((w, h))
        img_current = image
        img_previous = image
        img_Tk = ImageTk.PhotoImage(image)

        canvas.create_image(canvas_width/2, canvas_height/2, image=img_Tk)


def do_binary(canvas):
    print('do binary')
    global img_previous
    global img_current
    global img_Tk
    global img_grayscale

    img_previous = img_current

    img_current = img_current.convert('L')
    img_grayscale = img_current
    img_Tk = ImageTk.PhotoImage(img_current)
    canvas.create_image(canvas_width/2, canvas_height/2, image=img_Tk)


def do_threshold(canvas):
    print('do threshold')
    global img_current
    global img_previous
    global img_grayscale
    global img_threshold

    global img_Tk

    if img_grayscale is None or img_current is None:
        return
    img_ = img_current

    ret, img_ = cv2.threshold(np.array(img_), int(var_threshold.get()), 255, cv2.THRESH_BINARY)
    img_ = Image.fromarray(img_)
    img_threshold = img_
    img_previous = img_
    img_Tk = ImageTk.PhotoImage(img_)
    canvas.create_image(canvas_width / 2, canvas_height / 2, image=img_Tk)


def do_threshold_save():
    global img_threshold
    global img_current

    img_current = img_threshold


def do_denoise(canvas):

    global img_Tk
    global img_threshold
    global img_denoise
    global img_current
    global img_previous

    if img_current is None or img_Tk is None:
        return

    img_previous = img_current

    img_ = img_current

    # Denoise
    img_ = cv2.fastNlMeansDenoising(np.array(img_), None, 65, 7, 21)

    img_ = Image.fromarray(img_)
    img_denoise = img_
    img_current = img_
    img_Tk = ImageTk.PhotoImage(img_)
    canvas.create_image(canvas_width / 2, canvas_width / 2, image=img_Tk)
    print('do denoise')


def do_segment(canvas):

    global img_current

    global img_Tk

    global rectangles

    separation = 100.0
    if ent_Separation.get():
        separation = float(ent_Separation.get())

    print('separation: {}'.format(separation))
    if img_current is None:
        return
    print('do segment')

    img_ = np.array(img_current)

    im2, contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours len:{}".format(len(contours)))

    # remove small rectangles and combine all rectangles based on the relationship.
    # remove small and big rectangles.
    rects = [cv2.boundingRect(ctr) for ctr in contours]
    rects = [rect for rect in rects
             if rect[2] > 10 and rect[3] > 10 and rect[2] < img_.shape[0] and rect[3] < img_.shape[1]]

    # # combine rectangles based on the relationship: contain, intersect and disjoint.
    rects = combine_rectangles(rects, separation)

    for rect in rects:
        print("r0:{} r1:{} r2:{} r3:{}".format(rect[0], rect[1], rect[2], rect[3]))

        if rect[2] > 20 and rect[3] > 20:
            cv2.rectangle(img_, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

    # #
    print("rect len:{}".format(len(rects)))

    rectangles = rects
    img_ = Image.fromarray(img_)
    img_Tk = ImageTk.PhotoImage(img_)
    canvas.create_image(canvas_width / 2, canvas_width / 2, image=img_Tk)


def find_if_intersect(rect1, rect2):

    l = max(rect1[0], rect2[0])
    r = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
    b = max(rect1[1], rect2[1])
    t = min(rect1[1]+rect1[3], rect2[1]+rect2[3])
    minw = min(rect1[2], rect2[2])
    minh = min(rect1[3], rect2[3])

    if r-l > 0 and r-l < minw and t-b > 0 and t-b < minh:
        return True
    return False


def find_if_disjoint(rect1, rect2):

    return False


def combine_rectangles(rects, separation):
    if rects is None:
        return []
    print('rects len:{}'.format(len(rects)))

    # remove within rectangles
    within_cluster = list()
    for id1, r1 in enumerate(rects):
        x1, y1, w1, h1 = r1
        subcluster = list()
        for id2, r2 in enumerate(rects):
            x2, y2, w2, h2 = r2
            if x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2:
                subcluster.append(id2)
        within_cluster.append(subcluster)

    print(within_cluster)

    within_del_list = list()
    for wc in within_cluster:
        if wc:
            within_del_list.extend(wc)
    n_rects = list()
    for i in range(len(rects)):
        if i in within_del_list:
            continue
        else:
            n_rects.append(rects[i])
    rects = n_rects

    print(len(rects))
    # cluster
    cluster = list()
    rects_ = rects
    used_ids = set()
    for id1, r1 in enumerate(rects_):
        x1, y1, w1, h1 = r1
        subcluster = list()
        subcluster.append(id1)
        used_ids.add(id1)
        for id2, r2 in enumerate(rects_):
            if id2 in used_ids:
                continue
            x2, y2, w2, h2 = r2

            # intersection
            if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                subcluster.append(id2)
                used_ids.add(id2)
            # adjoin
            distance = math.sqrt(math.pow(x1 + w1 / 2 - x2 - w2 / 2, 2) + math.pow(y1 + h1 / 2 - y2 - h2 / 2, 2))
            if distance < separation:
                subcluster.append(id2)
                used_ids.add(id2)
        cluster.append(subcluster)

    print(cluster)

    # remove the cluster
    cluster_ = cluster
    cluster = list()
    for cls1 in cluster_:
        for cls2 in cluster_:
            if [x for x in cls1 if x in cls2]:
                cls1.extend(cls2)
        if set(cls1) not in cluster:
            cluster.append(set(cls1))

    print(cluster)
    new_rects = list()
    for clt in cluster:

        min_x = 100000
        min_y = 100000
        max_w = 0
        max_h = 0
        for c in clt:
            r_x, r_y, r_w, r_h = rects[c]
            min_x = min(min_x, r_x)
            min_y = min(min_y, r_y)
            max_w = max(max_w, r_x + r_w)
            max_h = max(max_h, r_y + r_h)

        new_x = min_x
        new_y = min_y
        new_w = max_w - min_x
        new_h = max_h - min_y
        new_rects.append((new_x, new_y, new_w, new_h))

    return new_rects


def get_distance(rect1, rect2):
    if rect1 is None or rect2 is None:
        return 0.0
    point1 = (rect1[0] + int(rect1[2]/2), rect1[1] + int(rect1[3]/2))
    point2 = (rect2[0] + int(rect2[2]/2), rect2[1] + int(rect2[3]/2))

    dist = math.sqrt((point1[0] - point2[0])*(point1[0] - point2[0]) + (point1[1] - point2[1])*(point1[1] - point2[1]))
    return dist


def calculate_mass(rect):
    global img_grayscale
    if rect is None:
        return 0, 0
    print(rect)
    ROWS = []
    COLS = []

    img_arr = np.array(img_grayscale)
    print(img_grayscale.size)
    print(img_arr.shape)

    for i in range(rect[2]):
        for j in range(rect[3]):
            if img_arr[rect[1]+j, rect[0]+i] < 127:

                ROWS.append(rect[0] + i)
                COLS.append(rect[1] + j)

    r = int(np.mean(ROWS))
    c = int(np.mean(COLS))

    return c, r


def do_save():
    global rectangles
    global original_image

    print('do save')
    if rectangles is None:
        print('Rectangles is None!')
        return
    print('rects lenght:{}'.format(len(rectangles)))
    # cut the original and save
    index = 0
    img_original_ = np.array(img_original)
    print(img_original_.shape)
    for rect in rectangles:
        startc = 0
        startr = 0
        endc = 0
        endr = 0
        mcr, mcc = calculate_mass(rect)
        if scale == 0.0:
            startc = rect[0]
            startr = rect[1]
            endc = rect[0] + rect[2]
            endr = rect[1] + rect[3]
        else:
            startc = int(rect[0] / scale)
            startr = int(rect[1] / scale)
            endc= int((rect[0]+rect[2]) / scale)
            endr = int((rect[1]+rect[3]) / scale)
            mcr = int(mcr / scale)
            mcc = int(mcc / scale)
        print("start:({},{}) end:({},{}) mc:({}, {})".format(startc, startr, endc, endr, mcr, mcc))

        # cut the region of original image and save
        if len(img_original_.shape) == 3:
            cut_region = img_original_[startr:endr, startc:endc, :]
        elif len(img_original_.shape) == 2:
            cut_region = img_original_[startr:endr, startc:endc]
        cv2.circle(cut_region, (mcc- startc, mcr - startr), 3, (0, 255, 0))

        # print(cut_region)
        if len(img_original_.shape) == 3:
            cut_region = cut_region[:, :, ::-1]
        elif len(img_original_.shape) == 2:
            cut_region = cut_region[:, :]

        cv2.imwrite('cut_{}.png'.format(index), cut_region)

        index += 1


while True:
    try:
        root.mainloop()
        break
    except UnicodeDecodeError:
        pass