#!/usr/bin/env python

import sys
import cv2

if __name__ == '__main__':
    # If image path and f/q is not passed as command line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    im = cv2.imread(sys.argv[1])
    # resize image
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))    

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()

    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    
    # number of region proposals to show
    numShowRects = 100
    increment = 50

    while True:
        imOut = im.copy()

        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        
    cv2.destroyAllWindows()
