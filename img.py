import cv2
from picamera2 import Picamera2
import time
cpt = 0
maxFrames = 30 # if you want 5 frames only.
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()



while cpt < maxFrames:
      im= picam2.capture_array()
      im=cv2.flip(im,-1)
      cv2.imwrite("/home/pi/images/A_%d.jpg" %cpt, im)
      time.sleep(0.5)
      cpt += 1

      
      cv2.imshow("im",im)
      key = cv2.waitKey(1)
      if cv2.waitKey == 27:  # esc
         break


cv2.destroyAllWindows()