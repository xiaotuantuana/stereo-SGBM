from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    Canvas,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    TOP,
    Button,
    Checkbutton,
    Entry,
    Frame,
    IntVar,
    Label,
    LabelFrame,
    OptionMenu,
    PanedWindow,
    Radiobutton,
    Scale,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
)

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "stereo_app_outputs"
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def find_default_video() -> Path:
    candidates = [
        APP_DIR / "car.avi",
        APP_DIR / "SGBM(Python)" / "car.avi",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for extension in VIDEO_EXTENSIONS:
        videos = sorted(APP_DIR.rglob(f"*{extension}"))
        if videos:
            return videos[0]
    return APP_DIR / "car.avi"


DEFAULT_VIDEO = find_default_video()


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        return False
    with path.open("wb") as file:
        file.write(buffer.tobytes())
    return True


@dataclass
class StereoCalibration:
    name: str
    size: tuple[int, int]
    left_camera_matrix: np.ndarray
    right_camera_matrix: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    R: np.ndarray
    T: np.ndarray

    def build_maps(self):
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_distortion,
            self.right_camera_matrix,
            self.right_distortion,
            self.size,
            self.R,
            self.T,
        )
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion, r1, p1, self.size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion, r2, p2, self.size, cv2.CV_16SC2
        )
        return left_map1, left_map2, right_map1, right_map2, q, roi1, roi2


def calibration_640x480() -> StereoCalibration:
    return StereoCalibration(
        name="640x480 原仓库 SGBM 标定",
        size=(640, 480),
        left_camera_matrix=np.array([[516.5066236, -1.444673028, 320.2950423], [0, 516.5816117, 270.7881873], [0, 0, 1.0]]),
        right_camera_matrix=np.array([[511.8428182, 1.295112628, 317.310253], [0, 513.0748795, 269.5885026], [0, 0, 1.0]]),
        left_distortion=np.array([[-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0]]),
        right_distortion=np.array([[-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0]]),
        R=np.array([[0.999911333, -0.004351508, 0.012585312], [0.004184066, 0.999902792, 0.013300386], [-0.012641965, -0.013246549, 0.999832341]]),
        T=np.array([-120.3559901, -0.188953775, -0.662073075]),
    )


def calibration_1280x720() -> StereoCalibration:
    return StereoCalibration(
        name="1280x720 原仓库 BM 标定",
        size=(1280, 720),
        left_camera_matrix=np.array([[986.4572391, 1.673607456, 651.0717611], [0, 1001.238398, 535.8195077], [0, 0, 1.0]]),
        right_camera_matrix=np.array([[998.5848065, 7.37746018, 667.3698587], [0, 1006.305891, 528.9731771], [0, 0, 1.0]]),
        left_distortion=np.array([[-0.154511565, 0.325173292, 0.006934081, 0.017466934, 0]]),
        right_distortion=np.array([[-0.192887524, 0.706728768, 0.004233541, 0.021340116, 0]]),
        R=np.array([[0.999925137, -0.003616734, -0.01168927], [0.003742452, 0.999935202, 0.010751105], [0.011649629, -0.010794046, 0.999873879]]),
        T=np.array([-117.3364039, 0.277054571, -3.7672413]),
    )


class StereoDepthEngine:
    def __init__(self):
        self.calibrations = [calibration_640x480(), calibration_1280x720()]
        self.cache: dict[tuple, tuple] = {}

    @staticmethod
    def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        half = w // 2
        return frame[:, :half].copy(), frame[:, half:half * 2].copy()

    def choose_calibration(self, half_size: tuple[int, int]) -> StereoCalibration | None:
        for cal in self.calibrations:
            if cal.size == half_size:
                return cal
        return None

    def resolve_calibration(self, half_size: tuple[int, int], settings: dict) -> tuple[StereoCalibration | None, str]:
        if settings.get("calibration_mode") == "手动填写":
            custom_calibration = settings.get("custom_calibration")
            if custom_calibration is None:
                return None, "手动标定参数无效"
            if custom_calibration.size != half_size:
                return None, f"手动标定尺寸 {custom_calibration.size[0]}x{custom_calibration.size[1]} 与当前单目尺寸 {half_size[0]}x{half_size[1]} 不一致"
            return custom_calibration, custom_calibration.name

        cal = self.choose_calibration(half_size)
        if cal is None:
            return None, "未校正/尺寸未匹配标定"
        return cal, cal.name

    def rectify(self, left: np.ndarray, right: np.ndarray, settings: dict):
        h, w = left.shape[:2]
        cal, cal_name = self.resolve_calibration((w, h), settings)
        if cal is None:
            q = self.fallback_q(w, h)
            return left, right, q, cal_name

        cache_key = (settings.get("calibration_mode", "内置匹配"), settings.get("calibration_signature", cal.name), w, h)
        if not settings["rectify"]:
            if cache_key not in self.cache:
                self.cache[cache_key] = cal.build_maps()
            q = self.cache[cache_key][4]
            return left, right, q, f"{cal.name} | 未执行校正"

        if cache_key not in self.cache:
            self.cache[cache_key] = cal.build_maps()
        left_map1, left_map2, right_map1, right_map2, q, _, _ = self.cache[cache_key]
        left_r = cv2.remap(left, left_map1, left_map2, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR)
        return left_r, right_r, q, cal_name

    @staticmethod
    def fallback_q(width: int, height: int) -> np.ndarray:
        focal = 700.0
        baseline = 120.0
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal], [0, 0, -1.0 / baseline, 0]], dtype=np.float32)

    @staticmethod
    def make_num_disparities(value: int) -> int:
        return max(16, int(value // 16) * 16)

    def compute(self, frame: np.ndarray, settings: dict):
        left_raw, right_raw = self.split_side_by_side(frame)
        return self.compute_pair(left_raw, right_raw, settings)

    def compute_pair(self, left_raw: np.ndarray, right_raw: np.ndarray, settings: dict):
        left, right, q, cal_name = self.rectify(left_raw, right_raw, settings)
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        method = settings["method"]
        block_size = int(settings["block_size"])
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        num_disp = self.make_num_disparities(int(settings["num_disparities"]))
        uniqueness = int(settings["uniqueness"])
        speckle_window = int(settings["speckle_window"])
        speckle_range = int(settings["speckle_range"])
        disp12 = int(settings["disp12"])

        start = time.perf_counter()
        if method == "BM":
            matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=max(5, block_size))
            matcher.setTextureThreshold(10)
            matcher.setUniquenessRatio(uniqueness)
            matcher.setSpeckleWindowSize(speckle_window)
            matcher.setSpeckleRange(speckle_range)
            matcher.setDisp12MaxDiff(disp12)
            raw = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        else:
            channels = 1
            p1 = int(settings["p1"]) if settings["p1"] > 0 else 8 * channels * block_size * block_size
            p2 = int(settings["p2"]) if settings["p2"] > 0 else 32 * channels * block_size * block_size
            matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=p1,
                P2=max(p2, p1 + 1),
                disp12MaxDiff=disp12,
                uniquenessRatio=uniqueness,
                speckleWindowSize=speckle_window,
                speckleRange=speckle_range,
                preFilterCap=31,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            if settings["wls"] and hasattr(cv2, "ximgproc"):
                right_matcher = cv2.ximgproc.createRightMatcher(matcher)
                disp_l = matcher.compute(gray_l, gray_r)
                disp_r = right_matcher.compute(gray_r, gray_l)
                wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
                wls.setLambda(float(settings["wls_lambda"]))
                wls.setSigmaColor(float(settings["wls_sigma"]) / 10.0)
                raw = wls.filter(disp_l, gray_l, None, disp_r).astype(np.float32) / 16.0
            else:
                raw = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0

        if settings["median"]:
            raw = cv2.medianBlur(raw.astype(np.float32), 5)
        if settings["improved_fill"]:
            raw = self.fill_small_holes(raw, gray_l, max_area=int(settings["hole_area"]))
        if settings["bilateral"]:
            filtered = cv2.bilateralFilter(raw.astype(np.float32), 5, 6, 3)
            filtered[raw <= 0] = 0
            raw = filtered

        elapsed = max(time.perf_counter() - start, 1e-6)
        points_3d = cv2.reprojectImageTo3D(raw.astype(np.float32), q, handleMissingValues=True)
        disparity_vis = self.colorize_disparity(raw)
        depth = self.disparity_to_depth(raw, q)
        depth_vis = self.colorize_depth(depth, float(settings["max_depth_m"]))
        return {
            "left": left,
            "right": right,
            "disparity": raw,
            "disparity_vis": disparity_vis,
            "depth_m": depth,
            "depth_vis": depth_vis,
            "points_3d": points_3d,
            "fps": 1.0 / elapsed,
            "elapsed_ms": elapsed * 1000.0,
            "calibration": cal_name,
        }

    @staticmethod
    def fill_small_holes(disparity: np.ndarray, guide: np.ndarray, max_area: int = 80) -> np.ndarray:
        invalid = disparity <= 0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), 8)
        fill_mask = np.zeros_like(invalid)
        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] <= max_area:
                fill_mask |= labels == label

        out = disparity.copy()
        h, w = out.shape
        for y in range(h):
            for x in np.where(fill_mask[y])[0]:
                candidates = []
                for direction in (-1, 1):
                    stop = -1 if direction < 0 else w
                    for xx in range(x + direction, stop, direction):
                        if out[y, xx] > 0 and not fill_mask[y, xx]:
                            candidates.append((abs(int(guide[y, xx]) - int(guide[y, x])), out[y, xx]))
                            break
                if candidates:
                    candidates.sort(key=lambda item: item[0])
                    out[y, x] = float(candidates[0][1])
        return out

    @staticmethod
    def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
        valid = disparity > 0
        if not np.any(valid):
            return np.zeros((*disparity.shape, 3), dtype=np.uint8)
        vmax = max(16.0, float(np.percentile(disparity[valid], 96)))
        norm = np.clip(disparity / vmax, 0, 1)
        img = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        color[~valid] = (0, 0, 0)
        return color

    @staticmethod
    def disparity_to_depth(disparity: np.ndarray, q: np.ndarray) -> np.ndarray:
        points = cv2.reprojectImageTo3D(disparity.astype(np.float32), q, handleMissingValues=True)
        z = points[:, :, 2].astype(np.float32)
        z[np.logical_or(disparity <= 0, ~np.isfinite(z))] = np.nan
        # Most calibration values in this repository are millimeters.
        if np.nanmedian(np.abs(z)) > 20:
            z = z / 1000.0
        z[z <= 0] = np.nan
        return z

    @staticmethod
    def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
        valid = np.isfinite(depth_m)
        if not np.any(valid):
            return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        clipped = np.clip(depth_m, 0, max_depth_m)
        norm = 1.0 - clipped / max(max_depth_m, 0.1)
        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        img = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        color[~valid] = (0, 0, 0)
        return color


class StereoDepthApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("纯视觉双目深度相机 - SGBM/BM 实时测距软件")
        self.root.geometry("1760x960")
        self.root.minsize(1360, 800)
        self.engine = StereoDepthEngine()
        self.capture: cv2.VideoCapture | None = None
        self.capture_right: cv2.VideoCapture | None = None
        self.preview_capture: cv2.VideoCapture | None = None
        self.preview_capture_right: cv2.VideoCapture | None = None
        self.running = False
        self.preview_running = False
        self.current_result: dict | None = None
        self.current_frame: np.ndarray | None = None
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.measurements: list[dict] = []
        self.frame_measurements: list[dict] = []
        self.confirmed_points: list[dict] = []
        self.pending_point: dict | None = None
        self.frame_index = 0
        self.resize_job = None
        self.preview_job = None
        self.initial_sash_placed = False
        self.depth_display_size = (1, 1)
        self.depth_display_offset = (0, 0)
        self.frame_queue: queue.Queue[tuple[np.ndarray | tuple[np.ndarray, np.ndarray], dict]] = queue.Queue(maxsize=2)
        self.capture_thread: threading.Thread | None = None

        self.source_type = StringVar(value="视频文件")
        self.video_path = StringVar(value=str(DEFAULT_VIDEO))
        self.camera_index = StringVar(value="0")
        self.camera_choice = StringVar(value="0 - 默认双目相机")
        self.camera_width = StringVar(value="2560")
        self.camera_height = StringVar(value="720")
        self.camera_fps_mode = StringVar(value="自适应")
        self.camera_fps = StringVar(value="30")
        self.preview_start_frame = StringVar(value="0")
        self.method = StringVar(value="SGBM")
        self.calibration_mode = StringVar(value="内置匹配")
        self.cal_width = StringVar(value="1280")
        self.cal_height = StringVar(value="720")
        self.left_fx = StringVar(value="986.4572391")
        self.left_fy = StringVar(value="1001.238398")
        self.left_cx = StringVar(value="651.0717611")
        self.left_cy = StringVar(value="535.8195077")
        self.right_fx = StringVar(value="998.5848065")
        self.right_fy = StringVar(value="1006.305891")
        self.right_cx = StringVar(value="667.3698587")
        self.right_cy = StringVar(value="528.9731771")
        self.left_distortion = StringVar(value="-0.154511565, 0.325173292, 0.006934081, 0.017466934, 0")
        self.right_distortion = StringVar(value="-0.192887524, 0.706728768, 0.004233541, 0.021340116, 0")
        self.rotation_values = StringVar(value="0.999925137, -0.003616734, -0.01168927, 0.003742452, 0.999935202, 0.010751105, 0.011649629, -0.010794046, 0.999873879")
        self.translation_values = StringVar(value="-117.3364039, 0.277054571, -3.7672413")
        self.rectify = IntVar(value=1)
        self.wls = IntVar(value=0)
        self.median = IntVar(value=0)
        self.bilateral = IntVar(value=1)
        self.improved_fill = IntVar(value=1)

        self.block_size = IntVar(value=3)
        self.num_disparities = IntVar(value=64)
        self.p1 = IntVar(value=0)
        self.p2 = IntVar(value=0)
        self.uniqueness = IntVar(value=15)
        self.speckle_window = IntVar(value=150)
        self.speckle_range = IntVar(value=2)
        self.disp12 = IntVar(value=1)
        self.wls_lambda = IntVar(value=8000)
        self.wls_sigma = IntVar(value=15)
        self.hole_area = IntVar(value=80)
        self.max_depth_m = IntVar(value=10)

        self.status = StringVar(value="未启动")
        self.measure_text = StringVar(value="点击深度图查看距离")
        self._build_ui()
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        top = Frame(self.root)
        top.pack(side=TOP, fill="x", padx=8, pady=6)

        source_box = LabelFrame(top, text="双目输入源")
        source_box.pack(side=LEFT, padx=4)
        source_box.grid_columnconfigure(1, weight=1)
        Radiobutton(source_box, text="视频", variable=self.source_type, value="视频文件").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        Entry(source_box, textvariable=self.video_path, width=48).grid(row=0, column=1, columnspan=4, padx=4, pady=2, sticky="ew")
        Button(source_box, text="选择视频", command=self.choose_video).grid(row=0, column=5, padx=4)
        Label(source_box, text="预览起始帧").grid(row=0, column=6, padx=(8, 2), sticky="e")
        Entry(source_box, textvariable=self.preview_start_frame, width=8).grid(row=0, column=7, padx=(2, 4), pady=2, sticky="w")
        Radiobutton(source_box, text="双目相机", variable=self.source_type, value="双目相机").grid(row=1, column=0, padx=4, pady=2, sticky="w")
        self.camera_option = OptionMenu(source_box, self.camera_choice, self.camera_choice.get(), command=self.on_camera_choice)
        self.camera_option.grid(row=1, column=1, padx=4, pady=2, sticky="ew")
        Button(source_box, text="识别相机", command=self.refresh_cameras).grid(row=1, column=2, padx=4)
        Label(source_box, text="采集分辨率").grid(row=1, column=3, padx=(4, 2), sticky="e")
        Entry(source_box, textvariable=self.camera_width, width=7).grid(row=1, column=4, padx=(2, 2), pady=2, sticky="w")
        Label(source_box, text="x").grid(row=1, column=5, padx=2, sticky="w")
        Entry(source_box, textvariable=self.camera_height, width=7).grid(row=1, column=6, padx=(2, 4), pady=2, sticky="w")
        Label(source_box, text="帧率").grid(row=1, column=7, padx=(6, 2), sticky="e")
        OptionMenu(source_box, self.camera_fps_mode, "自适应", "手动").grid(row=1, column=8, padx=(2, 2), pady=2, sticky="ew")
        Entry(source_box, textvariable=self.camera_fps, width=6).grid(row=1, column=9, padx=(2, 4), pady=2, sticky="w")

        action_box = LabelFrame(top, text="控制")
        action_box.pack(side=LEFT, padx=4)
        self.start_button = Button(action_box, text="启动", width=9, command=self.start)
        self.start_button.grid(row=0, column=0, padx=4, pady=4)
        Button(action_box, text="停止", width=9, command=self.stop).grid(row=0, column=1, padx=4)
        Button(action_box, text="预览选点", width=10, command=self.preview_first_frame).grid(row=0, column=2, padx=4)
        Button(action_box, text="截图/保存", width=10, command=self.save_snapshot).grid(row=0, column=3, padx=4)
        Button(action_box, text="导出逐帧CSV", width=12, command=self.export_measurements).grid(row=0, column=4, padx=4)
        Button(action_box, text="确定测点", width=10, command=self.confirm_pending_point).grid(row=1, column=0, padx=4, pady=4)
        Button(action_box, text="清空测点", width=10, command=self.clear_points).grid(row=1, column=1, padx=4, pady=4)
        Button(action_box, text="暂停选点", width=10, command=self.pause_preview).grid(row=1, column=2, padx=4, pady=4)

        self.main_paned = PanedWindow(self.root, orient=HORIZONTAL, sashwidth=10, sashrelief="raised")
        self.main_paned.pack(side=TOP, fill=BOTH, expand=True, padx=8, pady=4)

        self.image_paned = PanedWindow(self.main_paned, orient=HORIZONTAL, sashwidth=8, sashrelief="raised")
        left_panel = Frame(self.image_paned)
        self.left_label = Label(left_panel, text="左图", bg="#111", fg="white")
        self.left_label.pack(side=TOP, fill=BOTH, expand=True, padx=4, pady=4)
        self.depth_label = Label(left_panel, text="深度图/点击测距", bg="#111", fg="white")
        self.depth_label.pack(side=TOP, fill=BOTH, expand=True, padx=4, pady=4)
        self.depth_label.bind("<Button-1>", self.on_depth_click)

        right_panel = Frame(self.image_paned)
        self.right_label = Label(right_panel, text="右图", bg="#111", fg="white")
        self.right_label.pack(side=TOP, fill=BOTH, expand=True, padx=4, pady=4)
        self.disp_label = Label(right_panel, text="视差图", bg="#111", fg="white")
        self.disp_label.pack(side=TOP, fill=BOTH, expand=True, padx=4, pady=4)
        self.image_paned.add(left_panel, minsize=420)
        self.image_paned.add(right_panel, minsize=420)

        control_outer = Frame(self.main_paned, width=520)
        self._build_scrollable_controls(control_outer)
        self.main_paned.add(self.image_paned, minsize=760, stretch="always")
        self.main_paned.add(control_outer, minsize=460, width=520, stretch="never")
        self.root.after(150, self.place_initial_sash)

        bottom = Frame(self.root)
        bottom.pack(side=BOTTOM, fill="x", padx=8, pady=6)
        Label(bottom, textvariable=self.status, anchor="w").pack(side=LEFT, fill="x", expand=True)
        Label(bottom, textvariable=self.measure_text, anchor="e", fg="#0b5").pack(side=RIGHT)

    def _build_scrollable_controls(self, parent: Frame):
        canvas = Canvas(parent, width=500, highlightthickness=0)
        scrollbar = Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = Frame(canvas)
        scroll_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))

        def _wheel(event):
            if event.num == 5 or event.delta < 0:
                canvas.yview_scroll(3, "units")
            elif event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-3, "units")

        canvas.bind_all("<MouseWheel>", _wheel)
        canvas.bind_all("<Button-4>", _wheel)
        canvas.bind_all("<Button-5>", _wheel)
        self._build_algorithm_controls(scroll_frame)

    def _build_algorithm_controls(self, parent: Frame):
        algo = LabelFrame(parent, text="算法")
        algo.pack(fill="x", pady=4)
        algo.grid_columnconfigure(0, weight=1)
        algo.grid_columnconfigure(1, weight=1)
        Label(algo, text="匹配方法", anchor="w").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        OptionMenu(algo, self.method, "SGBM", "BM").grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        Checkbutton(algo, text="立体校正", variable=self.rectify).grid(row=1, column=0, padx=4, sticky="w")
        Checkbutton(algo, text="WLS", variable=self.wls).grid(row=1, column=1, padx=4, sticky="w")
        Checkbutton(algo, text="中值滤波", variable=self.median).grid(row=2, column=0, padx=4, sticky="w")
        Checkbutton(algo, text="小孔洞填充", variable=self.improved_fill).grid(row=2, column=1, padx=4, sticky="w")
        Checkbutton(algo, text="双边保边", variable=self.bilateral).grid(row=3, column=0, padx=4, sticky="w")

        calibration = LabelFrame(parent, text="相机标定参数")
        calibration.pack(fill="x", pady=4)
        calibration.grid_columnconfigure(0, weight=1)
        calibration.grid_columnconfigure(1, weight=1)
        Label(calibration, text="标定来源", anchor="w").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        OptionMenu(calibration, self.calibration_mode, "内置匹配", "手动填写").grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self._entry(calibration, "单目宽", self.cal_width, 1, 0)
        self._entry(calibration, "单目高", self.cal_height, 1, 1)
        self._entry(calibration, "左 fx", self.left_fx, 2, 0)
        self._entry(calibration, "左 fy", self.left_fy, 2, 1)
        self._entry(calibration, "左 cx", self.left_cx, 3, 0)
        self._entry(calibration, "左 cy", self.left_cy, 3, 1)
        self._entry(calibration, "右 fx", self.right_fx, 4, 0)
        self._entry(calibration, "右 fy", self.right_fy, 4, 1)
        self._entry(calibration, "右 cx", self.right_cx, 5, 0)
        self._entry(calibration, "右 cy", self.right_cy, 5, 1)
        self._wide_entry(calibration, "左畸变 k1,k2,p1,p2,k3", self.left_distortion, 6)
        self._wide_entry(calibration, "右畸变 k1,k2,p1,p2,k3", self.right_distortion, 7)
        self._wide_entry(calibration, "旋转矩阵 R（9个数）", self.rotation_values, 8)
        self._wide_entry(calibration, "平移向量 T（3个数，通常mm）", self.translation_values, 9)

        params = LabelFrame(parent, text="SGBM/BM 参数")
        params.pack(fill="x", pady=4)
        params.grid_columnconfigure(0, weight=1)
        params.grid_columnconfigure(1, weight=1)
        self._scale(params, "blockSize", self.block_size, 3, 15, 2, 0, 0)
        self._scale(params, "numDisparities", self.num_disparities, 16, 256, 16, 0, 1)
        self._scale(params, "P1=0自动", self.p1, 0, 3000, 50, 1, 0)
        self._scale(params, "P2=0自动", self.p2, 0, 12000, 100, 1, 1)
        self._scale(params, "uniquenessRatio", self.uniqueness, 0, 30, 1, 2, 0)
        self._scale(params, "speckleWindow", self.speckle_window, 0, 300, 10, 2, 1)
        self._scale(params, "speckleRange", self.speckle_range, 0, 64, 1, 3, 0)
        self._scale(params, "disp12MaxDiff", self.disp12, -1, 10, 1, 3, 1)

        post = LabelFrame(parent, text="后处理/显示")
        post.pack(fill="x", pady=4)
        post.grid_columnconfigure(0, weight=1)
        post.grid_columnconfigure(1, weight=1)
        self._scale(post, "WLS lambda", self.wls_lambda, 1000, 20000, 500, 0, 0)
        self._scale(post, "WLS sigma x0.1", self.wls_sigma, 5, 30, 1, 0, 1)
        self._scale(post, "填充最大面积", self.hole_area, 10, 500, 10, 1, 0)
        self._scale(post, "最大深度(m)", self.max_depth_m, 1, 30, 1, 1, 1)

        info = LabelFrame(parent, text="功能说明")
        info.pack(fill="both", expand=True, pady=4)
        Label(
            info,
            justify=LEFT,
            anchor="nw",
            text=(
                "1. 输入支持左右拼接视频或 USB 双目相机。\n"
                "2. 建议先点“预览选点”，确定测点后再启动。\n"
                "3. 截图会保存左图、右图、视差图、深度图和参数。\n"
                "4. 导出逐帧CSV会保存每一帧每个测点的距离。\n"
                "5. 标定来源选“手动填写”后，将使用上方参数进行校正和深度换算。\n"
                "6. 若单目尺寸与标定尺寸不一致，会自动提示尺寸不匹配。"
            ),
        ).pack(fill="both", expand=True, padx=4, pady=4)

        points = LabelFrame(parent, text="测点列表")
        points.pack(fill="both", expand=True, pady=4)
        self.points_text = Text(points, height=10, width=58, state=DISABLED)
        self.points_text.pack(fill="both", expand=True, padx=4, pady=4)

    @staticmethod
    def _entry(parent: Frame, label: str, var: StringVar, row: int, column: int):
        cell = Frame(parent)
        cell.grid(row=row, column=column, padx=4, pady=2, sticky="ew")
        cell.grid_columnconfigure(1, weight=1)
        Label(cell, text=label, width=8, anchor="w").grid(row=0, column=0, sticky="w")
        Entry(cell, textvariable=var, width=12).grid(row=0, column=1, sticky="ew")

    @staticmethod
    def _wide_entry(parent: Frame, label: str, var: StringVar, row: int):
        Label(parent, text=label, anchor="w").grid(row=row, column=0, padx=4, pady=(4, 0), sticky="w")
        Entry(parent, textvariable=var).grid(row=row, column=1, padx=4, pady=(4, 0), sticky="ew")

    @staticmethod
    def _scale(parent: Frame, label: str, var: IntVar, start: int, end: int, resolution: int, row: int, column: int):
        cell = Frame(parent)
        cell.grid(row=row, column=column, padx=4, pady=2, sticky="ew")
        cell.grid_columnconfigure(0, weight=1)
        Label(cell, text=label, anchor="w").grid(row=0, column=0, sticky="ew")
        Scale(cell, variable=var, from_=start, to=end, resolution=resolution, orient=HORIZONTAL, length=210).grid(row=1, column=0, sticky="ew")

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="选择左右拼接双目视频",
            initialdir=str(APP_DIR),
            filetypes=[("AVI 双目视频", "*.avi"), ("Video", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.video_path.set(path)
            self.source_type.set("视频文件")

    @staticmethod
    def resolve_video_path(source: str) -> Path:
        path = Path(source.strip().strip('"'))
        if path.is_absolute():
            return path
        return APP_DIR / path

    @staticmethod
    def open_video_capture(path: Path) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            return capture
        capture.release()

        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if capture.isOpened():
            return capture
        capture.release()
        return capture

    def on_camera_choice(self, choice: str):
        index = choice.split(" ", 1)[0].strip()
        if index.isdigit():
            self.camera_index.set(index)
        self.source_type.set("双目相机")

    def refresh_cameras(self):
        camera_indices = self.detect_cameras()
        if not camera_indices:
            devices = ["0 - 单目相机 0"]
            self.status.set("未识别到可打开的相机，保留默认相机 0")
        else:
            if len(camera_indices) >= 2:
                left_index, right_index = camera_indices[0], camera_indices[1]
                devices = [f"{left_index} + {right_index} - 双目相机 左{left_index}/右{right_index}"]
                devices.extend(f"{index} - 单目相机 {index}" for index in camera_indices)
            else:
                devices = [f"{camera_indices[0]} - 单目相机 {camera_indices[0]}"]
            self.status.set(f"已识别 {len(camera_indices)} 个相机接口")

        menu = self.camera_option["menu"]
        menu.delete(0, "end")
        for choice in devices:
            menu.add_command(label=choice, command=lambda value=choice: self.set_camera_choice(value))
        self.set_camera_choice(devices[0])

    def set_camera_choice(self, choice: str):
        self.camera_choice.set(choice)
        self.on_camera_choice(choice)

    def selected_camera_indices(self) -> list[int]:
        prefix = self.camera_choice.get().split("-", 1)[0]
        normalized = prefix.replace("+", " ").replace(",", " ")
        indices: list[int] = []
        for token in normalized.split():
            if token.isdigit():
                indices.append(int(token))
        if indices:
            return indices[:2]
        try:
            return [int(self.camera_index.get())]
        except ValueError:
            return []

    @staticmethod
    def detect_cameras(max_index: int = 10) -> list[int]:
        devices: list[int] = []
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        for index in range(max_index):
            capture = cv2.VideoCapture(index, backend)
            opened = capture.isOpened()
            if opened:
                devices.append(index)
            capture.release()
        return devices

    @staticmethod
    def _parse_float_list(text: str, expected: int, label: str) -> list[float]:
        normalized = text.replace("，", ",").replace("；", ",").replace(";", ",").replace("\n", ",")
        values = [part.strip() for part in normalized.replace(" ", ",").split(",") if part.strip()]
        if len(values) != expected:
            raise ValueError(f"{label}需要 {expected} 个数，当前为 {len(values)} 个")
        try:
            return [float(value) for value in values]
        except ValueError as exc:
            raise ValueError(f"{label}包含无法识别的数字") from exc

    @staticmethod
    def _parse_int_value(text: str, label: str) -> int:
        try:
            value = int(float(text.strip()))
        except ValueError as exc:
            raise ValueError(f"{label}必须是数字") from exc
        if value <= 0:
            raise ValueError(f"{label}必须大于 0")
        return value

    @staticmethod
    def _parse_float_value(text: str, label: str) -> float:
        try:
            return float(text.strip())
        except ValueError as exc:
            raise ValueError(f"{label}必须是数字") from exc

    def build_custom_calibration(self) -> StereoCalibration:
        width = self._parse_int_value(self.cal_width.get(), "单目宽")
        height = self._parse_int_value(self.cal_height.get(), "单目高")
        left_fx = self._parse_float_value(self.left_fx.get(), "左 fx")
        left_fy = self._parse_float_value(self.left_fy.get(), "左 fy")
        left_cx = self._parse_float_value(self.left_cx.get(), "左 cx")
        left_cy = self._parse_float_value(self.left_cy.get(), "左 cy")
        right_fx = self._parse_float_value(self.right_fx.get(), "右 fx")
        right_fy = self._parse_float_value(self.right_fy.get(), "右 fy")
        right_cx = self._parse_float_value(self.right_cx.get(), "右 cx")
        right_cy = self._parse_float_value(self.right_cy.get(), "右 cy")
        left_distortion = np.array([self._parse_float_list(self.left_distortion.get(), 5, "左畸变")], dtype=np.float64)
        right_distortion = np.array([self._parse_float_list(self.right_distortion.get(), 5, "右畸变")], dtype=np.float64)
        rotation = np.array(self._parse_float_list(self.rotation_values.get(), 9, "旋转矩阵R"), dtype=np.float64).reshape(3, 3)
        translation = np.array(self._parse_float_list(self.translation_values.get(), 3, "平移向量T"), dtype=np.float64)
        left_camera_matrix = np.array([[left_fx, 0, left_cx], [0, left_fy, left_cy], [0, 0, 1.0]], dtype=np.float64)
        right_camera_matrix = np.array([[right_fx, 0, right_cx], [0, right_fy, right_cy], [0, 0, 1.0]], dtype=np.float64)
        return StereoCalibration(
            name=f"手动标定 {width}x{height}",
            size=(width, height),
            left_camera_matrix=left_camera_matrix,
            right_camera_matrix=right_camera_matrix,
            left_distortion=left_distortion,
            right_distortion=right_distortion,
            R=rotation,
            T=translation,
        )

    def calibration_signature(self) -> str:
        values = [
            self.cal_width.get(),
            self.cal_height.get(),
            self.left_fx.get(),
            self.left_fy.get(),
            self.left_cx.get(),
            self.left_cy.get(),
            self.right_fx.get(),
            self.right_fy.get(),
            self.right_cx.get(),
            self.right_cy.get(),
            self.left_distortion.get(),
            self.right_distortion.get(),
            self.rotation_values.get(),
            self.translation_values.get(),
        ]
        return "|".join(values)

    def open_capture_source(self, preview: bool = False) -> cv2.VideoCapture | None:
        self.capture_right = None
        self.preview_capture_right = None
        if self.source_type.get() == "视频文件":
            video_path = self.resolve_video_path(self.video_path.get())
            if not video_path.exists():
                messagebox.showerror("错误", f"视频不存在：{video_path}")
                return None
            capture = self.open_video_capture(video_path)
        else:
            camera_indices = self.selected_camera_indices()
            if not camera_indices:
                messagebox.showerror("错误", "请选择一个可用相机")
                return None
            try:
                capture_width = self._parse_int_value(self.camera_width.get(), "采集宽度")
                capture_height = self._parse_int_value(self.camera_height.get(), "采集高度")
                if self.camera_fps_mode.get() == "手动":
                    capture_fps = self._parse_float_value(self.camera_fps.get(), "相机帧率")
                    if capture_fps <= 0:
                        raise ValueError("相机帧率必须大于 0")
                else:
                    capture_fps = None
            except ValueError as exc:
                messagebox.showerror("错误", str(exc))
                return None

            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
            capture = cv2.VideoCapture(camera_indices[0], backend)
            self.configure_camera_capture(capture, capture_width, capture_height, capture_fps)
            if len(camera_indices) >= 2:
                right_capture = cv2.VideoCapture(camera_indices[1], backend)
                self.configure_camera_capture(right_capture, capture_width, capture_height, capture_fps)
                if preview:
                    self.preview_capture_right = right_capture
                else:
                    self.capture_right = right_capture

        if not capture.isOpened():
            self.release_extra_capture(preview)
            if self.source_type.get() == "视频文件":
                messagebox.showerror("错误", f"无法打开视频：{video_path}\n请确认 AVI 文件未损坏，且当前 OpenCV 支持该视频编码。")
            else:
                messagebox.showerror("错误", "无法打开输入源")
            return None
        if self.source_type.get() == "双目相机":
            right_capture = self.preview_capture_right if preview else self.capture_right
            if right_capture is not None and not right_capture.isOpened():
                capture.release()
                self.release_extra_capture(preview)
                messagebox.showerror("错误", "无法打开右相机输入源")
                return None
            if right_capture is None:
                self.status.set("仅识别/选择到 1 个相机：只显示左图，不进行左右分割和双目测距")
            else:
                self.status.set("已打开 2 个相机：左/右相机将进行双目测距")
        if preview and self.source_type.get() == "视频文件":
            try:
                start_frame = max(0, int(float(self.preview_start_frame.get())))
            except ValueError:
                capture.release()
                messagebox.showerror("错误", "预览起始帧必须是 0 或更大的整数")
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        return capture

    @staticmethod
    def configure_camera_capture(capture: cv2.VideoCapture, width: int, height: int, fps: float | None):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, fps)

    def release_extra_capture(self, preview: bool):
        right_capture = self.preview_capture_right if preview else self.capture_right
        if right_capture is not None:
            right_capture.release()
        if preview:
            self.preview_capture_right = None
        else:
            self.capture_right = None

    def preview_first_frame(self):
        if self.running:
            messagebox.showinfo("提示", "请先停止播放，再预览选点")
            return
        self.stop_preview(silent=True)
        self.preview_capture = self.open_capture_source(preview=True)
        if self.preview_capture is None:
            return
        self.preview_running = True
        self.status.set("预览播放中 | 到目标画面后点击“暂停选点”，再在深度图上选择测点")
        self.preview_loop()

    def preview_loop(self):
        if not self.preview_running or self.preview_capture is None:
            return
        ok, frame = self.preview_capture.read()
        if not ok:
            self.stop_preview(silent=True)
            self.status.set("预览已到末尾，请调整预览起始帧后重新预览")
            return
        payload: np.ndarray | tuple[np.ndarray, np.ndarray] = frame
        if self.preview_capture_right is not None:
            ok_right, right_frame = self.preview_capture_right.read()
            if not ok_right:
                self.stop_preview(silent=True)
                self.status.set("右相机预览读取失败")
                return
            payload = (frame, right_frame)
        metadata = self.capture_metadata(self.preview_capture)
        try:
            result = self.build_result(payload, metadata)
        except Exception as exc:
            self.stop_preview(silent=True)
            messagebox.showerror("错误", f"预览处理失败：{exc}")
            return
        self.current_frame = frame if isinstance(payload, np.ndarray) else payload[0]
        self.current_result = result
        self._show_images(result)
        if result.get("mode") == "mono":
            self.status.set(f"单目预览 | 帧 {metadata['frame_no']} | {metadata['display_time']} | 仅显示左图")
        else:
            self.status.set(
                f"预览播放 | 帧 {metadata['frame_no']} | {metadata['display_time']} | {result['calibration']} | 点击“暂停选点”后选点"
            )
        delay = self.preview_delay_ms()
        self.preview_job = self.root.after(delay, self.preview_loop)

    def preview_delay_ms(self) -> int:
        if self.preview_capture is None:
            return 30
        fps = float(self.preview_capture.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 1 or fps > 240:
            return 30
        return max(1, int(1000 / fps))

    def pause_preview(self):
        self.stop_preview(silent=False)

    def stop_preview(self, silent: bool = False):
        self.preview_running = False
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None
        if self.preview_capture_right is not None:
            self.preview_capture_right.release()
            self.preview_capture_right = None
        if not silent and self.current_result:
            if self.current_result.get("mode") == "mono":
                self.status.set(f"已暂停在帧 {self.current_result.get('frame_no', self.frame_index)} | 单目模式只能查看左图")
            else:
                self.status.set(
                    f"已暂停在帧 {self.current_result.get('frame_no', self.frame_index)} | 可在深度图上选点"
                )

    def capture_metadata(self, capture: cv2.VideoCapture) -> dict:
        frame_no = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or max(self.frame_index, 0))
        pos_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if self.source_type.get() == "视频文件":
            if pos_ms <= 0 and source_fps > 0 and frame_no > 0:
                pos_ms = frame_no * 1000.0 / source_fps
            display_time = self.format_milliseconds(pos_ms)
        else:
            display_time = datetime.now().strftime("%H:%M:%S")
        return {
            "frame_no": frame_no,
            "source_fps": source_fps,
            "display_time": display_time,
        }

    @staticmethod
    def format_milliseconds(milliseconds: float) -> str:
        total_seconds = max(0, int(milliseconds / 1000))
        ms = max(0, int(milliseconds % 1000))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
        return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

    def build_result(self, payload: np.ndarray | tuple[np.ndarray, np.ndarray], metadata: dict) -> dict:
        if isinstance(payload, tuple):
            result = self.engine.compute_pair(payload[0], payload[1], self.get_settings())
        elif self.source_type.get() == "双目相机":
            result = self.build_mono_result(payload)
        else:
            result = self.engine.compute(payload, self.get_settings())
        result.update(metadata)
        return result

    def build_mono_result(self, frame: np.ndarray) -> dict:
        start = time.perf_counter()
        depth = np.full(frame.shape[:2], np.nan, dtype=np.float32)
        disparity = np.zeros(frame.shape[:2], dtype=np.float32)
        points_3d = np.full((*frame.shape[:2], 3), np.nan, dtype=np.float32)
        elapsed = max(time.perf_counter() - start, 1e-6)
        right_placeholder = self.placeholder_image(frame.shape, "未启用右相机")
        disp_placeholder = self.placeholder_image(frame.shape, "单目模式无视差图")
        depth_placeholder = self.placeholder_image(frame.shape, "单目模式无深度图")
        return {
            "mode": "mono",
            "left": frame,
            "right": right_placeholder,
            "disparity": disparity,
            "disparity_vis": disp_placeholder,
            "depth_m": depth,
            "depth_vis": depth_placeholder,
            "points_3d": points_3d,
            "fps": 1.0 / elapsed,
            "elapsed_ms": elapsed * 1000.0,
            "calibration": "单目相机预览",
        }

    @staticmethod
    def placeholder_image(shape: tuple[int, ...], text: str) -> np.ndarray:
        h, w = shape[:2]
        image = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, min(w, h) / 850)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(image, text, (max((w - tw) // 2, 8), max((h + th) // 2, th + 8)), font, scale, (180, 220, 255), thickness, cv2.LINE_AA)
        return image

    def get_settings(self) -> dict:
        custom_calibration = None
        if self.calibration_mode.get() == "手动填写":
            custom_calibration = self.build_custom_calibration()
        return {
            "source_type": self.source_type.get(),
            "camera_index": self.camera_index.get(),
            "camera_choice": self.camera_choice.get(),
            "camera_width": self.camera_width.get(),
            "camera_height": self.camera_height.get(),
            "camera_fps_mode": self.camera_fps_mode.get(),
            "camera_fps": self.camera_fps.get(),
            "preview_start_frame": self.preview_start_frame.get(),
            "method": self.method.get(),
            "calibration_mode": self.calibration_mode.get(),
            "calibration_signature": self.calibration_signature(),
            "custom_calibration": custom_calibration,
            "rectify": bool(self.rectify.get()),
            "wls": bool(self.wls.get()),
            "median": bool(self.median.get()),
            "bilateral": bool(self.bilateral.get()),
            "improved_fill": bool(self.improved_fill.get()),
            "block_size": int(self.block_size.get()),
            "num_disparities": int(self.num_disparities.get()),
            "p1": int(self.p1.get()),
            "p2": int(self.p2.get()),
            "uniqueness": int(self.uniqueness.get()),
            "speckle_window": int(self.speckle_window.get()),
            "speckle_range": int(self.speckle_range.get()),
            "disp12": int(self.disp12.get()),
            "wls_lambda": int(self.wls_lambda.get()),
            "wls_sigma": int(self.wls_sigma.get()),
            "hole_area": int(self.hole_area.get()),
            "max_depth_m": int(self.max_depth_m.get()),
        }

    def start(self):
        self.stop_preview(silent=True)
        self.stop()
        self.capture = self.open_capture_source()
        if self.capture is None:
            return
        self.frame_index = 0
        self.frame_measurements.clear()
        self.measurements.clear()
        self.running = True
        self.start_button.configure(state=DISABLED)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.root.after(10, self._ui_loop)

    def stop(self):
        self.running = False
        self.start_button.configure(state=NORMAL)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.capture_right is not None:
            self.capture_right.release()
            self.capture_right = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self):
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                if self.source_type.get() == "视频文件" and self.capture is not None:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            payload: np.ndarray | tuple[np.ndarray, np.ndarray] = frame
            if self.capture_right is not None:
                ok_right, right_frame = self.capture_right.read()
                if not ok_right:
                    break
                payload = (frame, right_frame)
            metadata = self.capture_metadata(self.capture)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((payload, metadata))
        self.running = False

    def _ui_loop(self):
        if not self.running:
            self.start_button.configure(state=NORMAL)
            return
        try:
            payload, metadata = self.frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(10, self._ui_loop)
            return

        try:
            result = self.build_result(payload, metadata)
        except Exception as exc:
            self.status.set(f"处理失败：{exc}")
            self.root.after(30, self._ui_loop)
            return

        self.current_frame = payload if isinstance(payload, np.ndarray) else payload[0]
        self.current_result = result
        self.frame_index += 1
        if result.get("mode") != "mono":
            self.record_frame_measurements(result)
        self._show_images(result)
        if result.get("mode") == "mono":
            self.status.set(
                f"单目显示 | 帧 {result.get('frame_no', self.frame_index)} | {result.get('display_time', '')} | FPS {result['fps']:.1f} | 未进行双目测距"
            )
        else:
            self.status.set(
                f"帧 {result.get('frame_no', self.frame_index)} | {result.get('display_time', '')} | {result['calibration']} | FPS {result['fps']:.1f} | {result['elapsed_ms']:.1f} ms | "
                f"视差有效率 {np.mean(result['disparity'] > 0) * 100:.1f}% | 逐帧记录 {len(self.frame_measurements)} 行"
            )
        self.root.after(1, self._ui_loop)

    def _show_images(self, result: dict):
        if result.get("mode") == "mono":
            depth_with_points = result["depth_vis"].copy()
        else:
            depth_with_points = self.draw_measure_points(result["depth_vis"].copy())
        left_with_overlay = self.draw_runtime_overlay(result["left"].copy(), result)
        right_with_overlay = self.draw_runtime_overlay(result["right"].copy(), result)
        self.photo_refs = [
            self._set_image(self.left_label, left_with_overlay),
            self._set_image(self.right_label, right_with_overlay),
            self._set_image(self.disp_label, result["disparity_vis"]),
            self._set_image(self.depth_label, depth_with_points, remember_depth=True),
        ]

    @staticmethod
    def draw_runtime_overlay(image: np.ndarray, result: dict) -> np.ndarray:
        fps_text = f"FPS {float(result.get('fps', 0)):.1f}"
        time_text = f"Time {result.get('display_time', '--:--')}"
        StereoDepthApp._draw_corner_text(image, fps_text, top=True)
        StereoDepthApp._draw_corner_text(image, time_text, top=False)
        return image

    @staticmethod
    def _draw_corner_text(image: np.ndarray, text: str, top: bool):
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(w, h) / 900)
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(8, w - tw - 12)
        y = th + 10 if top else h - baseline - 10
        cv2.rectangle(image, (x - 6, y - th - 6), (min(w - 1, x + tw + 6), min(h - 1, y + baseline + 6)), (0, 0, 0), -1)
        cv2.putText(image, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def on_window_resize(self, event):
        if event.widget is not self.root:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.redraw_after_resize)

    def place_initial_sash(self):
        if self.initial_sash_placed or not hasattr(self, "main_paned"):
            return
        try:
            total_width = self.main_paned.winfo_width()
            if total_width > 1200:
                self.main_paned.sash_place(0, total_width - 530, 0)
                image_width = self.image_paned.winfo_width()
                if image_width > 900:
                    self.image_paned.sash_place(0, image_width // 2, 0)
                    self.initial_sash_placed = True
                else:
                    self.root.after(150, self.place_initial_sash)
            else:
                self.root.after(150, self.place_initial_sash)
        except Exception:
            pass

    def redraw_after_resize(self):
        self.resize_job = None
        if self.current_result:
            self._show_images(self.current_result)

    def _resize_for_label(self, label: Label, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(label.winfo_width() - 12, 320)
        max_h = max(label.winfo_height() - 12, 220)
        if label.winfo_width() <= 20 or label.winfo_height() <= 20:
            max_w, max_h = 620, 330
        scale = min(max_w / image.width, max_h / image.height)
        resized = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(resized, Image.Resampling.LANCZOS)
        return image, max_w, max_h

    def _set_image(self, label: Label, bgr: np.ndarray, remember_depth: bool = False) -> ImageTk.PhotoImage:
        image, max_w, max_h = self._resize_for_label(label, bgr)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, width=max_w, height=max_h)
        label.image_size = image.size
        if remember_depth:
            self.depth_display_size = image.size
            self.root.update_idletasks()
            self.depth_display_offset = (
                max((label.winfo_width() - image.size[0]) // 2, 0),
                max((label.winfo_height() - image.size[1]) // 2, 0),
            )
        return photo

    def draw_measure_points(self, image: np.ndarray) -> np.ndarray:
        for i, point in enumerate(self.confirmed_points, start=1):
            label = f"P{i}"
            if point.get("valid", False) and "distance_m" in point:
                label = f"P{i} {point['distance_m']:.2f}m"
            self._draw_marker(image, int(point["x"]), int(point["y"]), label, (0, 255, 255))
        if self.pending_point:
            if self.pending_point.get("valid", False) and "distance_m" in self.pending_point:
                label = f"P? {self.pending_point['distance_m']:.2f}m"
            else:
                label = "invalid"
            self._draw_marker(image, int(self.pending_point["x"]), int(self.pending_point["y"]), label, (0, 80, 255))
        return image

    @staticmethod
    def _draw_marker(image: np.ndarray, x: int, y: int, label: str, color: tuple[int, int, int]):
        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        cv2.circle(image, (x, y), 7, color, 2)
        tx = min(max(x + 8, 0), max(w - 150, 0))
        ty = min(max(y - 8, 18), h - 4)
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def on_depth_click(self, event):
        if not self.current_result:
            self.measure_text.set("请先点击“预览选点”，再在深度图上选择测点")
            return
        if self.current_result.get("mode") == "mono":
            self.measure_text.set("当前只有 1 个相机输入，只显示左图，不能进行双目深度测距")
            return
        depth_img = self.current_result["depth_vis"]
        h, w = depth_img.shape[:2]
        disp_w, disp_h = self.depth_display_size
        offset_x, offset_y = self.depth_display_offset
        x = int((event.x - offset_x) * w / max(disp_w, 1))
        y = int((event.y - offset_y) * h / max(disp_h, 1))
        if not (0 <= x < w and 0 <= y < h):
            return

        record = self.sample_point(self.current_result, x, y)
        self.pending_point = record
        self._show_images(self.current_result)
        if not record["valid"]:
            self.measure_text.set(f"待确认：像素({x},{y}) 当前深度无效，d={record['disparity']:.2f}")
            return
        self.measure_text.set(
            f"待确认：像素({x},{y}) | X={record['X_m']:.3f}m Y={record['Y_m']:.3f}m Z={record['Z_m']:.3f}m | "
            f"距离={record['distance_m']:.3f}m | d={record['disparity']:.2f}"
        )

    def sample_point(self, result: dict, x: int, y: int) -> dict:
        point = result["points_3d"][y, x].astype(float)
        z = result["depth_m"][y, x]
        disp = float(result["disparity"][y, x])
        record = {
            "x": int(x),
            "y": int(y),
            "valid": bool(np.isfinite(z) and disp > 0),
            "disparity": disp,
            "X_m": "",
            "Y_m": "",
            "Z_m": "",
            "distance_m": "",
        }
        if not record["valid"]:
            return record
        if np.nanmedian(np.abs(result["points_3d"][:, :, 2])) > 20:
            point = point / 1000.0
        record.update(
            {
                "X_m": float(point[0]),
                "Y_m": float(point[1]),
                "Z_m": float(point[2]),
                "distance_m": float(np.linalg.norm(point)),
            }
        )
        return record

    def confirm_pending_point(self):
        if not self.pending_point:
            messagebox.showinfo("提示", "请先在深度图上点击一个测点")
            return
        if not self.pending_point.get("valid", False):
            messagebox.showwarning("测点无效", "当前点击位置深度无效，请选择有颜色的有效深度区域")
            return
        record = dict(self.pending_point)
        record["id"] = len(self.confirmed_points) + 1
        self.confirmed_points.append(record)
        self.pending_point = None
        self.update_points_text()
        self.measure_text.set(
            f"已确认测点 P{record['id']}：({record['x']},{record['y']}) 距离={record['distance_m']:.3f}m"
        )

    def clear_points(self):
        self.confirmed_points.clear()
        self.pending_point = None
        self.frame_measurements.clear()
        self.measurements.clear()
        self.update_points_text()
        if self.current_result:
            self._show_images(self.current_result)
        self.measure_text.set("已清空测点和逐帧记录")

    def record_frame_measurements(self, result: dict):
        if not self.confirmed_points:
            return
        now = datetime.now().isoformat(timespec="milliseconds")
        for point in self.confirmed_points:
            sampled = self.sample_point(result, int(point["x"]), int(point["y"]))
            row = {
                "frame": self.frame_index,
                "time": now,
                "point_id": point["id"],
                "x": point["x"],
                "y": point["y"],
                "valid": sampled["valid"],
                "X_m": sampled["X_m"],
                "Y_m": sampled["Y_m"],
                "Z_m": sampled["Z_m"],
                "distance_m": sampled["distance_m"],
                "disparity": sampled["disparity"],
                "fps": float(result["fps"]),
                "elapsed_ms": float(result["elapsed_ms"]),
            }
            self.frame_measurements.append(row)
            point.update(sampled)
        self.measurements = self.frame_measurements
        self.update_points_text()

    def update_points_text(self):
        self.points_text.configure(state=NORMAL)
        self.points_text.delete("1.0", END)
        if not self.confirmed_points:
            self.points_text.insert(END, "暂无已确认测点。\n点击深度图后按“确定测点”。")
        else:
            for p in self.confirmed_points:
                if p.get("valid", False) and isinstance(p.get("distance_m"), (int, float)):
                    line = (
                        f"P{p['id']}  x={p['x']} y={p['y']}  "
                        f"Z={p['Z_m']:.3f}m  距离={p['distance_m']:.3f}m  d={p['disparity']:.2f}\n"
                    )
                else:
                    line = f"P{p['id']}  x={p['x']} y={p['y']}  当前帧无效  d={p.get('disparity', 0):.2f}\n"
                self.points_text.insert(END, line)
        self.points_text.configure(state=DISABLED)

    def save_snapshot(self):
        if not self.current_result:
            messagebox.showinfo("提示", "没有可保存的画面")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / timestamp
        out.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(out / "left.png", self.current_result["left"])
        imwrite_unicode(out / "right.png", self.current_result["right"])
        imwrite_unicode(out / "disparity_color.png", self.current_result["disparity_vis"])
        if self.current_result.get("mode") == "mono":
            depth_snapshot = self.current_result["depth_vis"].copy()
        else:
            depth_snapshot = self.draw_measure_points(self.current_result["depth_vis"].copy())
        imwrite_unicode(out / "depth_color.png", depth_snapshot)
        np.save(out / "disparity.npy", self.current_result["disparity"])
        np.save(out / "depth_m.npy", self.current_result["depth_m"])
        settings = self.get_settings()
        if isinstance(settings.get("custom_calibration"), StereoCalibration):
            calibration = settings["custom_calibration"]
            settings = dict(settings)
            settings["custom_calibration"] = {
                "name": calibration.name,
                "size": calibration.size,
                "left_camera_matrix": calibration.left_camera_matrix.tolist(),
                "right_camera_matrix": calibration.right_camera_matrix.tolist(),
                "left_distortion": calibration.left_distortion.tolist(),
                "right_distortion": calibration.right_distortion.tolist(),
                "R": calibration.R.tolist(),
                "T": calibration.T.tolist(),
            }
        (out / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "points.json").write_text(json.dumps(self.confirmed_points, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"已保存截图：{out}")

    def export_measurements(self):
        if not self.frame_measurements:
            messagebox.showinfo("提示", "还没有逐帧测距数据：请先预览选点、确定测点，然后启动播放一段时间")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fieldnames = [
            "frame",
            "time",
            "point_id",
            "x",
            "y",
            "valid",
            "X_m",
            "Y_m",
            "Z_m",
            "distance_m",
            "disparity",
            "fps",
            "elapsed_ms",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frame_measurements)
        self.status.set(f"已导出逐帧测距数据：{path}")

    def on_close(self):
        self.stop_preview(silent=True)
        self.stop()
        self.root.destroy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Tk()
    app = StereoDepthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
