import cv2
import math

image_file = '/Users/liupeng/Documents/dl2tcc/IP/scan0008.jpg'

img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

width, height, channel = img.shape

if img is None:
    print('image is none!')

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

print('contours len:{}'.format(len(contours)))

separation = 120

rects = list()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20 or w >= width or h>= height:
        continue
    rects.append((x, y, w, h))

# remove within rectangles
within_cluster = list()
for id1, r1 in enumerate(rects):
    x1, y1, w1, h1 = r1
    subcluster = list()
    for id2, r2 in enumerate(rects):
        x2, y2, w2, h2 = r2
        if x1 < x2 and y1 < y2 and x1+w1 > x2+w2 and y1+h1 > y2+h2:
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
        if x1 < x2+w2 and x1+w1 > x2 and y1 < y2+h2 and y1+h1 > y2:
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
    new_x = 0
    new_y = 0
    new_w = 0
    new_h = 0

    min_x = 100000
    min_y = 100000
    max_w = 0
    max_h = 0
    for c in clt:
        r_x, r_y, r_w, r_h = rects[c]
        min_x = min(min_x, r_x)
        min_y = min(min_y, r_y)
        max_w = max(max_w, r_x+r_w)
        max_h = max(max_h, r_y+r_h)

    new_x = min_x
    new_y = min_y
    new_w = max_w - min_x
    new_h = max_h - min_y
    new_rects.append((new_x, new_y, new_w, new_h))


for r in new_rects:
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)


# cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imwrite('contours.jpg', img)
cv2.imwrite('threshold.jpg', threshed_img)
