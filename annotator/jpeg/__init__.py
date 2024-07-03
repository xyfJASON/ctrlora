import cv2


class JpegCompressor:
    def __call__(self, img, jpeg_quality):
        # make jpeg artifacts to img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, encimg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
