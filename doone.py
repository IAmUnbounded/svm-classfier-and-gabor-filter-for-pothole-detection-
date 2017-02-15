
 
import numpy as np
import cv2
from cv2 import ml
def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		l=3
		while (l<48):
			kern=cv2.getGaborKernel((ksize, ksize), 4.0, theta,l, 0.5, 0, ktype=cv2.CV_32F)
			l = l+10
			#print l
			filters.append(kern)
	return filters
 
def process(img, filters):
	accum=[]
 	for kern in filters:
 		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 		accum.append(fimg)
 	return accum
def men(img):
	sum=0
	sum1=0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			sum=sum+int(img[i,j])
			cal=int(img[i,j])*int(img[i,j])
			sum1=sum1+cal
		val=img.shape[0]*img.shape[1]
		sum=sum/val
		sum1=sum1/val
	return (sum,sum1)
	
if __name__ == '__main__':
 import sys
 
 print __doc__
 
 img = cv2.imread("/home/sukhad/Downloads/Potholes-in-Paisley.jpg")
 img1 = cv2.imread("/home/sukhad/Downloads/road-dawn-mountains-sky.jpeg")

 if img is None:
 	print 'Failed to load image file:'
 	sys.exit(1)
 if img1 is None:
 	print 'Failed to load image file:'
 	sys.exit(1)
 	
 Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 Img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
 filters = build_filters()
 
 res1 = process(Img, filters)
 res2=process(Img1,filters)
 fea=[]
 labels=[]
 labels.append(1)
 labels.append(1)
 labels.append(1)
 labels.append(0)
 labels.append(0)
 fea.append([])
 fea.append([])
 fea.append([])
 fea.append([])
 fea.append([])
 
 for feature in res1:
 	m,e=men(feature)
 	fea[0].append(m)
 	fea[0].append(e)
 img = cv2.imread("/home/sukhad/Downloads/pothole.jpg")
 Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 res1 = process(Img, filters)
 for feature in res1:
 	m,e=men(feature)
 	fea[1].append(m)
 	fea[1].append(e)
 img = cv2.imread("/home/sukhad/Downloads/Potholes+022515.jpg")
 Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 res1 = process(Img, filters)
 for feature in res1:
 	m,e=men(feature)
 	fea[2].append(m)
 	fea[2].append(e)

 for feature in res2:
 	m,e=men(feature)
 	fea[3].append(m)
 	fea[3].append(e)
 img1 = cv2.imread("/home/sukhad/Downloads/road-dawn-mountains-sky.jpeg")
 Img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
 res2=process(Img1,filters)
 for feature in res2:
 	m,e=men(feature)
 	fea[4].append(m)
 	fea[4].append(e)
 training_data = np.array(fea, dtype = np.float32)
 
 responses = np.array(labels, dtype = np.int32)
 svm = cv2.ml.SVM_create()
 svm.setType(cv2.ml.SVM_C_SVC)
 svm.setKernel(cv2.ml.SVM_LINEAR)

 svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
 svm.train(training_data, cv2.ml.ROW_SAMPLE, responses)
 svm.save('svm_data.dat')
 re=[]
 re.append([])
 re.append([])
 img1 = cv2.imread("/home/sukhad/Downloads/road-dawn-mountains-sky.jpeg")
 Img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
 res2=process(Img1,filters)
 for feature in res2:
 	m,e=men(feature)
 	re[0].append(m)
 	re[0].append(e)
 img1 = cv2.imread("/home/sukhad/Desktop/road-sky-sand-street.jpg")
 Img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
 res2=process(Img1,filters)
 for feature in res2:
 	m,e=men(feature)
 	re[1].append(m)
 	re[1].append(e)
 
 test = np.array(re, dtype = np.float32)
 result=svm.predict(test)
 print result

 

 cv2.waitKey(0)
 