# dataLoader/colmapUtils/run_colmap.py
import os
import subprocess
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["COLMAP_NO_GUI"] = "1"

def run_colmap_cli(image_dir, work_dir, camera_model=None, camera_params=None, verbose=True, skip_if_exists=True):
    """
    Chạy COLMAP CLI pipeline: feature_extractor -> exhaustive_matcher -> mapper -> image_undistorter
    - image_dir: folder chứa ảnh input
    - work_dir: folder sẽ chứa database.db, sparse/, dense/
    - camera_model: (optional) e.g. "PINHOLE", "SIMPLE_RADIAL", ...
    - camera_params: (optional) string hoặc list cho --ImageReader.camera_params
    - skip_if_exists: nếu True và work_dir/sparse/0 tồn tại -> sẽ bỏ qua
    Trả về: đường dẫn tới sparse folder (work_dir/sparse)
    """
    os.makedirs(work_dir, exist_ok=True)
    database_path = os.path.join(work_dir, "database.db")
    sparse_dir = os.path.join(work_dir, "sparse")
    undistort_dir = os.path.join(work_dir, "dense")

    # nếu đã có kết quả thì không chạy lại
    
    if skip_if_exists and os.path.exists(os.path.join(sparse_dir, "0")):
        if verbose:
            print(f"[run_colmap_cli] sparse/0 already exists in {sparse_dir}, skip.")
        return sparse_dir

    def run(cmd):
        # if verbose:
        #     print("RUN:", " ".join(cmd))
        print("[DEBUG] Running command:", " ".join(cmd))

        subprocess.run(cmd, check=True)

    # build commands
    cmd_feat = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",   # ✅ bật GPU
        "--ImageReader.camera_model", "PINHOLE"  # ✅ DJI ảnh nên dùng PINHOLE
    ]

    if camera_params:
        if isinstance(camera_params, (list, tuple)):
            for p in camera_params:
                cmd_feat += ["--ImageReader.camera_params", str(p)]
        else:
            cmd_feat += ["--ImageReader.camera_params", str(camera_params)]

    cmd_match = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "1"   # ✅ bật GPU
    ]



    os.makedirs(sparse_dir, exist_ok=True)
    cmd_mapper = ["colmap", "mapper", "--database_path", database_path, "--image_path", image_dir, "--output_path", sparse_dir]

    # undistort / prepare dense folder (chỉ chạy nếu mapper tạo được sparse/0)
    cmd_undistort = ["colmap", "image_undistorter", "--image_path", image_dir, "--input_path", os.path.join(sparse_dir, "0"), "--output_path", undistort_dir, "--output_type", "COLMAP"]

    try:
        run(cmd_feat)
        run(cmd_match)
        run(cmd_mapper)
    except subprocess.CalledProcessError as e:
        print("[run_colmap_cli] ERROR running COLMAP command:", e)
        raise

    if os.path.exists(os.path.join(sparse_dir, "0")):
        try:
            run(cmd_undistort)
        except subprocess.CalledProcessError:
            print("[run_colmap_cli] image_undistorter failed — but mapper succeeded. You can try undistort manually.")
    else:
        print("[run_colmap_cli] Warning: mapper did not produce sparse/0 — check COLMAP stdout.")

    if verbose:
        print(f"[run_colmap_cli] finished. sparse folder: {sparse_dir}")
    return sparse_dir
