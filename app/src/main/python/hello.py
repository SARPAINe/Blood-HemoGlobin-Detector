import cv2
import numpy as np
import copy
import imutils
folc= np.zeros((800,500))
# print(type(folc))
ext=(0,0)
extr=(0,0)
b=[]
def process_image(img,cnt=0):
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 40
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


    # -- Read imager
    # img = cv2.imread('whole_hand.jpg')
    # img = cv2.resize(img, (500, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # cv2.imshow('edges',edges)
    edges = cv2.dilate(edges, None)
    # cv2.imshow('dilate',edges)
    edges = cv2.erode(edges, None)
    # cv2.imshow('erode',edges)

    # -- Find contours in edges, sort by area
    contour_info = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    l = str(len(contours))
    # print("Number of contours = " + l)
    # print(cv2.contourArea(contours[0]))
    area = 0
    loc = -1
    for i in range(int(l)):
        if cv2.contourArea(contours[i]) > area:
            area = cv2.contourArea(contours[i])
            loc = i
    cv2.drawContours(img, contours, loc, (0, 0, 255), 3)
    # cv2.imshow('contour', img)


    c=contours[loc]
    mask = np.zeros(edges.shape)
    # cv2.fillConvexPoly(mask, c, (255,255,255))
    cv2.drawContours(mask,contours,loc,(255,255,255),-1)


    # cv2.imshow('mask', mask)

    """for c in contours:
        cv2.fillConvexPoly(mask, c, (255))"""

    # -- Smooth mask, then blur it
    # mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)

    if(cnt==1):
        mask = cv2.erode(mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=5)

    mask_stack = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    # cv2.imshow("mask2",mask)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
    # cv2.imshow("mask_stack",mask_stack)


    # -- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    # cv2.imshow('img.astype', img)
    masked = (mask_stack * img)
    # cv2.imshow('masked1', masked)
    # masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)

    masked = (masked * 255).astype('uint8')

    # cv2.imshow('masked', masked)                                   # Display
    # cv2.waitKey()

    c = contours[loc]
    # print(c)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # print("cX:")
    # print(cX)
    # print("cY:")
    # print(cY)
    # draw the contour and center of the shape on the image
    # cv2.drawContours( masked, -1, (0, 255, 0), 2)
    far_point = (cX, cY)

    cv2.circle(masked, far_point, 7, (255, 255, 255), -1)
    cv2.putText(masked, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image
    # cv2.imshow("masked", masked)
    # cv2.waitKey()



    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    global ext
    ext=extTop
    global extr
    extr=extRight
    # print(type(extTop))

    #extBot = tuple(c[c[:, :, 1].argmax()][0])
    # print(cv2.pointPolygonTest(c,extTop,False))

    """print(extLeft)
    print(extTop)
    print(extRight)"""

    # print('extTop x:',extTop[0],' extTop y:',extTop[1])
    # cv2.circle(masked, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(masked, extRight, 8, (0, 255, 0), -1)
    cv2.circle(masked, extTop, 8, (255, 0, 0), -1)
    #cv2.circle(masked, extBot, 8, (255, 255, 0), -1)
    #b,g,r=cv2.split(img);
    #img=cv2.merge((b,g,r))
    # show the output image
    # cv2.imshow("Image", masked)
    # cv2.waitKey(0)
    list=[extTop[0],extTop[1],extTop[0],extTop[1]]

    lo=list[0]
    g=[list[0],0,0,0,0]
    for i in range(list[0],500):
        if(cv2.pointPolygonTest(c, (i,list[1]+10), False)==0):
            g[1]=i
            break
    # print('g:',g)
    for i in range(list[0],500):
        if(cv2.pointPolygonTest(c, (i,list[1]+20), False)==0):
            g[2]=i
            break
    # print('g:',g)
    for i in range(list[0],500):
        if(cv2.pointPolygonTest(c, (i,list[1]+30), False)==0):
            g[3]=i
            break
    # print('g:',g)
    for i in range(list[0],500):
        if(cv2.pointPolygonTest(c, (i,list[1]+40), False)==0):
            g[4]=i
            break
    # print('g:',g)

    endpoint=(g[2],list[1]+40)
    g=[list[0],0,0,0,0]
    for i in range(list[0],0,-1):
        if(cv2.pointPolygonTest(c, (i,list[1]+10), False)==0):
            g[1]=i
            break
    # print('g:',g)
    for i in range(list[0],0,-1):
        if(cv2.pointPolygonTest(c, (i,list[1]+20), False)==0):
            g[2]=i
            break
    # print('g:',g)
    for i in range(list[0],0,-1):
        if(cv2.pointPolygonTest(c, (i,list[1]+30), False)==0):
            g[3]=i
            break
    # print('g:',g)
    for i in range(list[0],0,-1):
        if(cv2.pointPolygonTest(c, (i,list[1]+40), False)==0):
            g[4]=i
            break
    # print('g:',g)



    list=[extTop[0],extTop[1],extTop[0],extTop[1]]
    startpoint=(g[2],list[1])
    # startpoint=(g[0]-int((g[4]-g[0])*.6),list[1])
    bt=find_break(startpoint[0],endpoint[0],startpoint[1])
    b.append(bt)
    bf=endpoint[0]
    # print("bf:",bf," bt:",bt)
    cv2.rectangle(masked,startpoint,(bf,bt),(0,0,255),2)

    # print('final x:',list[0],' final y:',list[1])
    # cv2.imshow('masked',masked)
    global folc
    folc =masked
    # print("start point:", startpoint)
    # print("end point:", endpoint)
    return (startpoint,endpoint)


    #cv2.imwrite("background_removed.jpg",masked)"""

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    # return h, s, v
    return v

def x_find_break(topx,topy,btm):
    mid=int((topy+btm)/2)
    mn1=-1
    mn2=800
    for j in range(mid-10,mid+10):
        for i in range(topx,topx-50,-1):
            if(i>=500 or j>=800):
                break
            b, g, r = img[j, i]
            if(i<topx):
                b, g, r = img[j, i]
                x=rgb_to_hsv(r,g,b)
                b, g, r = img[j, i+1]
                y=rgb_to_hsv(r,g,b)
                if(abs(x-y)>=3):
                    mn1=max(mn1,i)
                    # cv2.circle(img, (i, j), 1, (0, 0, 255), -1)
    for j in range(mid-10,mid+10):
        for i in range(topx,topx+50):
            if(i>=500 or j>=800):
                break;
            b, g, r = img[j, i]
            if(i>topx):
                b, g, r = img[j, i]
                x=rgb_to_hsv(r,g,b)
                b, g, r = img[j, i-1]
                y=rgb_to_hsv(r,g,b)
                if(abs(x-y)>=3):
                    mn2=min(mn2,i)
                    # cv2.circle(img, (i, j), 1, (0, 255, 0), -1)
    return mn1,mn2

def find_break(ra,rb,plus):
    global img
    mn=800
    for i in range(ra,rb):
        if(i==500):
            break
        for j in range(41+plus,71+plus):
            if(j>=800):
                break
            b, g, r = img[j, i]
            if(j>(41+plus)):
                b, g, r = img[j, i]
                x=rgb_to_hsv(r,g,b)
                b, g, r = img[j-1, i]
                y=rgb_to_hsv(r,g,b)
                if(abs(x-y)>=2):
                    mn=min(mn,j)
                    # cv2.circle(img, (i, j), 1, (255, 0, 0), -1)
    return mn
img = np.zeros((800,500,3),dtype=np.uint8)
def test(red,blue,green):
    mask = np.zeros((800, 500, 3), dtype=np.uint8)
    for i in range(0, 500):
        for j in range(0, 800):
            aa = blue[i][j]
            bb = green[i][j]
            cc = red[i][j]
            mask[j, i] = (aa, bb, cc);


    mask = cv2.resize(mask, (500, 800))
    global img
    img=mask

    output_image=copy.copy(img)

    first_coor=process_image(img,1)
    img2=folc
    first_peak=ext

 # print(first_coor[0][0]," ",first_coor[0][1]," ",first_coor[1][0]," ",first_coor[1][1])
    left=[int(first_coor[0][0]-(first_coor[1][0]-first_coor[0][0])*.4)]
    right=[int(first_coor[1][0]+(first_coor[1][0]-first_coor[0][0])*.4)]
    top=[int(first_coor[0][1])]
    bottom=[extr[1]]
    # print(left[0],right[0],top[0],bottom[0])
    cv2.rectangle(img2, (left[0],top[0]), (right[0],bottom[0]), (0, 0, 0), -1)
    # cv2.imshow("img1",img2)

    second_coor=process_image(img2)
    img2=folc
    second_peak=ext
    # cv2.imshow("returnedImage2",img2)
    if(second_peak[0]<first_peak[0]):
        left.append(int(second_coor[0][0]-(second_coor[1][0]-second_coor[0][0])*.7))
        right.append(int(second_coor[1][0]+(second_coor[1][0]-second_coor[0][0])*.4))
        top.append(int(first_coor[0][1]))
        bottom.append(extr[1])
        cv2.rectangle(img2, (left[1],top[1]), (right[1],bottom[1]), (0, 0, 0), -1)

        # cv2.imshow("img2",img2)
        third_coor = process_image(img2)
        img2 = folc
        third_peak = ext
        # cv2.imshow("returnedImage3",img2)
        left.append(int(third_coor[0][0] - (third_coor[1][0] - third_coor[0][0]) * .7))
        right.append(int(third_coor[1][0] + (extr[0] - third_coor[0][0]) - (third_coor[1][0] - third_coor[0][0])))
        top.append(int(first_coor[0][1]))
        bottom.append(extr[1])
        cv2.rectangle(img2, (left[2], top[2]), (right[2], bottom[2]), (0, 0, 0), -1)
        # cv2.imshow("img3",img2)

    else:
        left.append(int(second_coor[0][0] - (second_coor[1][0] - second_coor[0][0]) * .7))
        right.append(int(second_coor[1][0] + (second_coor[1][0] - second_coor[0][0]) *.4 ))
        top.append(int(first_coor[0][1]))
        bottom.append(extr[1])
        cv2.rectangle(img2, (left[1], top[1]), (right[1], bottom[1]), (0, 0, 0), -1)

        third_coor = process_image(img2)
        img2 = folc
        third_peak = ext
        # cv2.imshow("returnedImage3",img2)
        left.append(int(third_coor[0][0] - (third_coor[1][0] - third_coor[0][0]) * .9))
        right.append(int(third_coor[1][0] + (extr[0] - third_coor[0][0]) - (third_coor[1][0] - third_coor[0][0])))
        top.append(int(first_coor[0][1]))
        bottom.append(extr[1])
        cv2.rectangle(img2, (left[2], top[2]), (right[2], bottom[2]), (0, 0, 0), -1)
        # cv2.imshow("img3",img2)

    # cv2.imshow("img2",img2)


    fourth_coor=process_image(img2)
    fourth_peak=ext
    """drawing top points of fingers
    """
    # print(first_peak)
    # print(second_peak)
    # print(third_peak)
    # print(fourth_peak)
    # print(extr)

    # cv2.circle(output_image, first_peak, 8, (255, 0, 0), -1)
    l, r = x_find_break(first_peak[0],first_peak[1],b[0])
    t=first_peak[1]+10
    cv2.rectangle(output_image,(l,t),(r,b[0]),(0,0,255),1)

# cv2.circle(output_image, second_peak, 8, (255, 0, 0), -1)
    l,r=x_find_break(second_peak[0],second_peak[1],b[1])
    t=second_peak[1]+10
    cv2.rectangle(output_image,(l,t),(r,b[1]),(0,0,255),1)

    # cv2.circle(output_image, third_peak, 8, (255, 0, 0), -1)
    l,r=x_find_break(third_peak[0],third_peak[1],b[2])
    t=third_peak[1]+10
    cv2.rectangle(output_image,(l,t),(r,b[2]),(0,0,255),1)

    # cv2.circle(output_image, fourth_peak, 8, (255, 0, 0), -1)
    l,r=x_find_break(fourth_peak[0],fourth_peak[1],b[3])
    t=fourth_peak[1]+10
    cv2.rectangle(output_image,(l,t),(r,b[3]),(0,0,255),1)

# cv2.circle(output_image, extr, 8, (255, 0, 0), -1)
    extrb=find_break(extr[0]-40,extr[0],extr[1])
    b.append(extrb)
    l,r=x_find_break(extr[0]-20,extr[1],b[4])
    t=extr[1]+10
    cv2.rectangle(output_image,(l-10,t),(r,b[4]),(0,0,255),1)


    l, r = x_find_break(first_peak[0],first_peak[1],b[0])
    t=first_peak[1]+10
    cv2.rectangle(output_image,(l,t),(r,b[0]),(0,0,255),1)

    blue, green, red = cv2.split(output_image)
    return red, green, blue

