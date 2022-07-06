import cv2
from torchvision import transforms

import model

# transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         ])

studyimage=cv2.imread(r'D:\Project\Gaze estimation\datatest\Image\p00/left/1.jpg')

imagepath=studyimage
print(imagepath)

net=model.EGNet(imagepath)

print(net)


