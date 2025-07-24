"""
Capture 2D images from the Zivid camera.
"""
from datetime import timedelta
import zivid
import cv2

def _set_sampling_pixel(settings: zivid.Settings, camera: zivid.Camera) -> None:
    """Get sampling pixel setting based on the camera model.

    Args:
        settings: Zivid settings instance
        camera: Zivid camera instance

    Raises:
        ValueError: If the camera model is not supported

    """
    if (
        camera.info.model is zivid.CameraInfo.Model.zividTwo
        or camera.info.model is zivid.CameraInfo.Model.zividTwoL100
        or camera.info.model is zivid.CameraInfo.Model.zivid2PlusM130
        or camera.info.model is zivid.CameraInfo.Model.zivid2PlusM60
        or camera.info.model is zivid.CameraInfo.Model.zivid2PlusL110
    ):
        settings.sampling.pixel = zivid.Settings.Sampling.Pixel.blueSubsample2x2
    elif (
        camera.info.model is zivid.CameraInfo.Model.zivid2PlusMR130
        or camera.info.model is zivid.CameraInfo.Model.zivid2PlusMR60
        or camera.info.model is zivid.CameraInfo.Model.zivid2PlusLR110
    ):
        settings.sampling.pixel = zivid.Settings.Sampling.Pixel.by2x2
    else:
        raise ValueError(f"Unhandled enum value {camera.info.model}")


def _main() -> None:
    app = zivid.Application()

    print("Connecting to camera")
    # real camera
    camera = app.connect_camera()

    # File camera 
    # file_camera = "C:/ProgramData/Zivid/FileCameraZivid2PlusMR60.zfc"
    # camera = app.create_file_camera(file_camera)

    print("Configuring 2D settings")
    settings = zivid.Settings()
    settings_2d = zivid.Settings2D()

    settings_2d.sampling.color = zivid.Settings2D.Sampling.Color.rgb
    settings_2d.sampling.pixel = zivid.Settings2D.Sampling.Pixel.all

    settings_2d.processing.color.balance.red = 1.0
    settings_2d.processing.color.balance.blue = 1.0
    settings_2d.processing.color.balance.green = 1.0
    settings_2d.processing.color.gamma = 1.0

    settings_2d.processing.color.experimental.mode = zivid.Settings2D.Processing.Color.Experimental.Mode.automatic

    settings_2d.acquisitions.append(
        zivid.Settings2D.Acquisition(
            aperture=4.0,
            exposure_time=timedelta(microseconds=1677),
            brightness=2.5,
            gain=1.0,
        )
    )

    settings.color = settings_2d
    _set_sampling_pixel(settings, camera)

    print("Capturing 2D frame")
    print("Getting RGBA image")
    while True: 
        with  camera.capture_2d(settings) as frame_2d:
            # image_srgb = frame_2d.image_rgba_srgb() # this is rgba, we need bgr.
            # bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            bgr = frame_2d.image_bgra_srgb().copy_data()
            dst = cv2.resize(bgr, dsize=(972, 600), interpolation=cv2.INTER_AREA) # 1944 x 1200

            cv2.imshow('Live-Stream',dst)
            if cv2.waitKey(20) & 0xFF == 27:
                break        


if __name__ == "__main__":
    _main()