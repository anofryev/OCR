import cv2


def get_surf_descriptors(img):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 100, # default:100
                                      nOctaves=4,  # default: 4
                                      nOctaveLayers=3, # default: 3
                                      extended=False, # default: False
                                      upright=False) # default: False
    kp, des = surf.detectAndCompute(img,None)
    
    return kp, des

def get_sift_descriptors(img):
    sift = cv2.SIFT_create(nfeatures=0,
                           nOctaveLayers=3,
                           contrastThreshold=0.04,
                           edgeThreshold=10,
                           sigma=1.6)
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def get_star_brief_descriptors(img):
    star = cv2.xfeatures2d.StarDetector_create(maxSize = 13, #45 
                                               responseThreshold=15,#30
                                               lineThresholdProjected=20,#10 best 20
                                               lineThresholdBinarized=8, #8 
                                               suppressNonmaxSize=5)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img, None)
    kp, des = brief.compute(img, kp)

    return kp, des
    
def get_orb_descriptors(img):
    orb = cv2.ORB_create(nfeatures = 200,
                        scaleFactor = 1.5,#1.2
                        nlevels = 8,
                        edgeThreshold = 31,
                        firstLevel = 0,
                        WTA_K = 2,
                        patchSize = 31,
                        fastThreshold = 20)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    
    return kp, des

# Initiate BRIEF extractor
#brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
#kp = star.detect(img,None)
# compute the descriptors with BRIEF
#kp, des = brief.compute(img, kp)
