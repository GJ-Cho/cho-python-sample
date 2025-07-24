"""
Stitch two point clouds using a transformation estimated by Local Point Cloud Registration and apply Voxel Downsample.

The ZDF files for this sample can be found in Zivid's Sample Data, under the main instructions for Zivid samples.
Zivid's Sample Data can be downloaded from  https://support.zivid.com/en/latest/api-reference/samples/sample-data.html.

Note: This example uses experimental SDK features, which may be modified, moved, or deleted in the future without notice.

"""

import zivid
from zivid.experimental.toolbox.point_cloud_registration import (
    LocalPointCloudRegistrationParameters,
    local_point_cloud_registration,
)
from zividsamples.display import display_pointcloud
from zividsamples.paths import get_sample_data_path
from pathlib import Path
from zivid.experimental.point_cloud_export import export_unorganized_point_cloud
from zivid.experimental.point_cloud_export.file_format import PLY


def _main() -> None:
    zivid.Application()

    # 파일 경로 리스트 생성
    base_dir = Path("C:/Zivid/python-test/stitching/shoes_02")  
    file_names = [f"0{i}.zdf" for i in range(1, 6)]
    frames = [zivid.Frame(base_dir / fname) for fname in file_names]

    # 포인트 클라우드 변환
    unorganized_point_clouds = [f.point_cloud().to_unorganized_point_cloud() for f in frames]

    # 스티칭 전 포인트 클라우드 모두 합쳐서 디스플레이
    print("Displaying point clouds before stitching")
    unorganized_not_stitched_point_cloud = zivid.UnorganizedPointCloud()
    for upc in unorganized_point_clouds:
        unorganized_not_stitched_point_cloud.extend(upc)
    # display_pointcloud(
    #     xyz=unorganized_not_stitched_point_cloud.copy_data("xyz"),
    #     rgb=unorganized_not_stitched_point_cloud.copy_data("rgba")[:, 0:3],
    # )

    print("Estimating transformation and stitching point clouds (누적 방식)")
    registration_params = LocalPointCloudRegistrationParameters()
    registration_params.max_iterations = 800
    registration_params.max_correspondence_distance = 15

    # 01과 02를 먼저 스티칭
    stitched_point_cloud = zivid.UnorganizedPointCloud()
    stitched_point_cloud.extend(unorganized_point_clouds[0])
    # registration은 원본으로 수행
    target_lpcr = unorganized_point_clouds[0]
    source = unorganized_point_clouds[1]
    source_lpcr = source
    local_point_cloud_registration_result = local_point_cloud_registration(
        target=target_lpcr, source=source_lpcr, parameters=registration_params
    )
    assert local_point_cloud_registration_result.converged(), "Registration for 02.zdf 실패"
    transform = local_point_cloud_registration_result.transform().to_matrix()
    source_transformed = source.transformed(transform)
    stitched_point_cloud.extend(source_transformed)
    print("Stitched 01.zdf with 02.zdf")
    # 다운샘플링 적용 (합친 후)
    stitched_point_cloud = stitched_point_cloud.voxel_downsampled(voxel_size=1.0, min_points_per_voxel=1)
    # 중간 결과 시각화
    display_pointcloud(
        xyz=stitched_point_cloud.copy_data("xyz"),
        rgb=stitched_point_cloud.copy_data("rgba")[:, 0:3],
    )

    # 이후 03, 04, 05를 누적된 결과와 계속 스티칭
    for i in range(2, 5):
        # registration은 다운샘플링 없이 누적 결과와 원본 source로 수행
        target_lpcr = stitched_point_cloud
        source = unorganized_point_clouds[i]
        source_lpcr = source
        local_point_cloud_registration_result = local_point_cloud_registration(
            target=target_lpcr, source=source_lpcr, parameters=registration_params
        )
        assert local_point_cloud_registration_result.converged(), f"Registration for 0{i+1}.zdf 실패"
        transform = local_point_cloud_registration_result.transform().to_matrix()
        print(f"Transform matrix for 0{i+1}.zdf:\n{transform}")
        source_transformed = source.transformed(transform)
        stitched_point_cloud.extend(source_transformed)
        print(f"Stitched 01~0{i}.zdf with 0{i+1}.zdf")
        # 다운샘플링 적용 (합친 후)
        stitched_point_cloud = stitched_point_cloud.voxel_downsampled(voxel_size=1.0, min_points_per_voxel=1)
        # 중간 결과 시각화
        display_pointcloud(
            xyz=stitched_point_cloud.copy_data("xyz"),
            rgb=stitched_point_cloud.copy_data("rgba")[:, 0:3],
        )

    print("Displaying point clouds after stitching")
    display_pointcloud(
        xyz=stitched_point_cloud.copy_data("xyz"),
        rgb=stitched_point_cloud.copy_data("rgba")[:, 0:3],
    )

    print("Voxel-downsampling the stitched point cloud")
    final_point_cloud = stitched_point_cloud.voxel_downsampled(voxel_size=0.5, min_points_per_voxel=1)
    display_pointcloud(
        xyz=final_point_cloud.copy_data("xyz"),
        rgb=final_point_cloud.copy_data("rgba")[:, 0:3],
    )

    print("01.zdf point count:", unorganized_point_clouds[0].copy_data("xyz").shape[0])
    print("stitched point count:", stitched_point_cloud.copy_data("xyz").shape[0])
    print("final (downsampled) point count:", final_point_cloud.copy_data("xyz").shape[0])

    # 최종 포인트 클라우드를 ply 파일로 저장
    file_name = Path(__file__).parent / "StitchedPointCloudOfShoes.ply"
    export_unorganized_point_cloud(final_point_cloud, PLY(str(file_name), layout=PLY.Layout.unordered))


if __name__ == "__main__":
    _main()
