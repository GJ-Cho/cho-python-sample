"""
Capture 2D images from the Zivid camera.
"""
import datetime
import zivid
import cv2

def _main() -> None:
    app = zivid.Application()

    print("Connecting to camera")
    # real camera
    camera = app.connect_camera()
    
    # File camera 
    # file_camera = "C:/ProgramData/Zivid/FileCameraZivid2PlusMR60.zfc"
    # camera = app.create_file_camera(file_camera)

    print("Configuring 2D settings")
    # Note: The Zivid SDK supports 2D captures with a single acquisition only
    settings_2d = zivid.Settings2D()
    settings_2d.acquisitions.append(zivid.Settings2D.Acquisition())
    settings_2d.acquisitions[0].exposure_time = datetime.timedelta(microseconds=1677)
    settings_2d.acquisitions[0].aperture = 4.0
    settings_2d.acquisitions[0].brightness = 0
    settings_2d.acquisitions[0].gain = 2
    settings_2d.processing.color.balance.red = 1.1
    settings_2d.processing.color.balance.green = 1.0
    settings_2d.processing.color.balance.blue = 1.5
    settings_2d.processing.color.gamma = 0.3

    print("Capturing 2D frame")
    print("Getting RGBA image")
    while True: 
        with camera.capture(settings_2d) as frame_2d:
            image = frame_2d.image_rgba()
            rgba = image.copy_data()

            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            dst = cv2.resize(bgr, dsize=(972, 600), interpolation=cv2.INTER_AREA) # 1944 x 1200

            cv2.imshow('Live-Stream',dst)
            if cv2.waitKey(20) & 0xFF == 27:
                break        


if __name__ == "__main__":
    _main()