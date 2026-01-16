import sys
import numpy as np
import os
import warnings
import threading
import mmap
import pickle
import time
from typing import Dict, List, Tuple, Union
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QGroupBox, QSpinBox,
    QDoubleSpinBox, QScrollArea, QPushButton, QProgressDialog
)
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import pyvista as pv
from pyvistaqt import QtInteractor
from concurrent.futures import ProcessPoolExecutor
import trimesh  # 需提前安装：pip install trimesh
# 导入Protobuf生成的模块（需与脚本同级目录）
import BVHData_pb2 as bvh_pb

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================== 全局配置 ======================
# 场景规模阈值（顶点数）：超过则视为大场景
LARGE_SCENE_THRESHOLD = 500000
# 缓存过期时间（7天）
CACHE_EXPIRE_DAYS = 7
# 并行进程数（限制最大8核，避免资源竞争）
MAX_WORKERS = min(os.cpu_count(), 8)
# Protobuf大文件阈值（100MB）
LARGE_PROTO_THRESHOLD = 1024 * 1024 * 100

# ====================== 全局样式表（不变） ======================
GROUP_BOX_STYLE = """
    QGroupBox {
        border: 1px solid #cccccc;
        border-radius: 6px;
        padding-top: 10px;
        margin-top: 8px;
        font-size: 12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 15px;
        padding: 0 8px 0 8px;
        color: #555555;
    }
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
    }
"""


# ====================== 坐标转换工具（不变） ======================
def convert_coordinate(coord: np.ndarray) -> np.ndarray:
    """坐标转换：Y↔Z互换（与C++保持一致）"""
    x, y, z = coord[0], coord[1], coord[2]
    return np.array([x, z, y])


# ====================== 缓存管理类（新增） ======================
class CacheManager:
    def __init__(self, cache_dir="Cache/VisualCache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}  # 内存缓存：key=场景标识，value=数据字典
        self.max_memory_scenes = 2  # 最多缓存2个场景

    def get_cache_key(self, obj_path: str, aabb_path: str) -> str:
        """生成缓存key：文件路径+修改时间戳（确保缓存有效性）"""
        obj_mtime = os.path.getmtime(obj_path) if os.path.exists(obj_path) else 0
        aabb_mtime = os.path.getmtime(aabb_path) if os.path.exists(aabb_path) else 0
        return f"{os.path.basename(obj_path)}_{obj_mtime}_{aabb_mtime}"

    def load_from_local_cache(self, obj_path: str, aabb_path: str) -> Union[Dict, None]:
        """从本地缓存加载预处理数据"""
        cache_key = self.get_cache_key(obj_path, aabb_path)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if not os.path.exists(cache_path):
            return None

        # 检查缓存是否过期
        cache_mtime = os.path.getmtime(cache_path)
        if time.time() - cache_mtime > CACHE_EXPIRE_DAYS * 24 * 3600:
            os.remove(cache_path)
            return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"本地缓存加载失败：{e}")
            os.remove(cache_path)
            return None

    def save_to_local_cache(self, obj_path: str, aabb_path: str, data: Dict):
        """保存预处理数据到本地缓存"""
        cache_key = self.get_cache_key(obj_path, aabb_path)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # 只保存核心预处理数据，不保存原始文件
        save_data = {
            "converted_vertices": data["converted_vertices"],
            "aabb_depth_nodes": data["aabb_depth_nodes"],
            "sphere_depth_nodes": data["sphere_depth_nodes"],
            "full_scene_mesh": data["full_scene_mesh"],
            "simplified_mesh": data["simplified_mesh"],
            "scene_extent": data["scene_extent"],
            "scene_center": data["scene_center"]
        }

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"本地缓存保存失败：{e}")

    def add_memory_cache(self, obj_path: str, aabb_path: str, data: Dict):
        """添加到内存缓存（LRU淘汰）"""
        cache_key = self.get_cache_key(obj_path, aabb_path)
        if len(self.memory_cache) >= self.max_memory_scenes:
            # 淘汰最早的缓存
            oldest_key = next(iter(self.memory_cache.keys()))
            del self.memory_cache[oldest_key]
        self.memory_cache[cache_key] = data

    def get_memory_cache(self, obj_path: str, aabb_path: str) -> Union[Dict, None]:
        """从内存缓存获取数据"""
        cache_key = self.get_cache_key(obj_path, aabb_path)
        return self.memory_cache.get(cache_key)


# ====================== 异步加载线程（新增） ======================
class AsyncDataLoader(QThread):
    progress_update = pyqtSignal(str, int)  # 进度描述、百分比
    load_finished = pyqtSignal(dict)  # 加载完成的数据字典
    load_failed = pyqtSignal(str)  # 加载失败原因

    def __init__(self, aabb_path: str, sphere_path: str, obj_path: str, ray_path: str):
        super().__init__()
        self.paths = (aabb_path, sphere_path, obj_path, ray_path)
        self.cache_manager = CacheManager()

    def run(self):
        try:
            aabb_path, sphere_path, obj_path, ray_path = self.paths
            data = {}

            # 1. 优先从缓存加载
            self.progress_update.emit("检查缓存有效性", 5)
            memory_cache = self.cache_manager.get_memory_cache(obj_path, aabb_path)
            if memory_cache:
                self.progress_update.emit("复用内存缓存", 90)
                data = memory_cache
                # 补充射线数据（缓存中不保存，避免占用空间）
                data["ray_data"] = self._load_proto(ray_path, bvh_pb.RayHitDataProto())
                self.progress_update.emit("加载完成", 100)
                self.load_finished.emit(data)
                return

            local_cache = self.cache_manager.load_from_local_cache(obj_path, aabb_path)
            if local_cache:
                self.progress_update.emit("复用本地缓存", 80)
                data = local_cache
                # 补充原始数据和射线数据
                data["aabb_bvh"] = self._load_proto(aabb_path, bvh_pb.BVHStructureProto())
                data["sphere_bvh"] = self._load_proto(sphere_path, bvh_pb.BVHStructureProto())
                data["ray_data"] = self._load_proto(ray_path, bvh_pb.RayHitDataProto())
                data["vertices"], data["faces"] = self._load_obj_fast(obj_path)
                self.cache_manager.add_memory_cache(obj_path, aabb_path, data)
                self.progress_update.emit("加载完成", 100)
                self.load_finished.emit(data)
                return

            # 2. 无缓存，执行快速加载
            self.progress_update.emit("解析AABB-BVH文件", 15)
            data["aabb_bvh"] = self._load_proto(aabb_path, bvh_pb.BVHStructureProto())

            self.progress_update.emit("解析Sphere-BVH文件", 30)
            data["sphere_bvh"] = self._load_proto(sphere_path, bvh_pb.BVHStructureProto())

            self.progress_update.emit("解析射线碰撞数据", 40)
            data["ray_data"] = self._load_proto(ray_path, bvh_pb.RayHitDataProto())

            self.progress_update.emit("解析OBJ模型", 50)
            data["vertices"], data["faces"] = self._load_obj_fast(obj_path)

            self.progress_update.emit("并行转换坐标", 60)
            data["converted_vertices"] = self._convert_coords_parallel(data["vertices"])

            self.progress_update.emit("分组BVH节点", 70)
            data["aabb_depth_nodes"] = self._group_by_depth(data["aabb_bvh"].nodes, is_aabb=True)
            data["sphere_depth_nodes"] = self._group_by_depth(data["sphere_bvh"].nodes, is_aabb=False)

            self.progress_update.emit("创建场景网格", 80)
            data["full_scene_mesh"] = self._create_mesh_fast(data["vertices"], data["faces"])
            data["simplified_mesh"] = self._create_simplified_mesh(data["vertices"], data["faces"])

            self.progress_update.emit("计算场景信息", 90)
            data["scene_extent"] = self._calculate_extent(data["converted_vertices"])
            data["scene_center"] = self._calculate_center(data["converted_vertices"])

            # 3. 保存到缓存
            self.cache_manager.save_to_local_cache(obj_path, aabb_path, data)
            self.cache_manager.add_memory_cache(obj_path, aabb_path, data)

            self.progress_update.emit("加载完成", 100)
            self.load_finished.emit(data)
        except Exception as e:
            self.load_failed.emit(f"加载失败：{str(e)}")

    def _load_proto(self, path: str, proto_msg) -> object:
        """优化Protobuf加载：大文件用内存映射"""
        if os.path.getsize(path) > LARGE_PROTO_THRESHOLD:
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    proto_msg.ParseFromString(mm.read())
        else:
            with open(path, "rb") as f:
                proto_msg.ParseFromString(f.read())
        return proto_msg

    def _load_obj_fast(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """优化OBJ加载：用trimesh替代原生解析"""
        mesh = trimesh.load(
            path,
            process=False,  # 禁用自动处理，仅解析
            skip_materials=True,  # 跳过材质（可视化不需要）
            skip_textures=True  # 跳过纹理（可视化不需要）
        )
        return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

    def _convert_coords_parallel(self, vertices: np.ndarray) -> np.ndarray:
        """并行处理坐标转换"""
        if len(vertices) < LARGE_SCENE_THRESHOLD:
            # 小场景单线程更快（避免进程切换开销）
            return np.array([convert_coordinate(v) for v in vertices])

        # 大场景分片并行
        chunks = np.array_split(vertices, MAX_WORKERS)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(self._convert_chunk, chunks)
        return np.concatenate(list(results))

    @staticmethod
    def _convert_chunk(chunk: np.ndarray) -> np.ndarray:
        """单个分片的坐标转换（子进程执行）"""
        return np.array([convert_coordinate(v) for v in chunk])

    def _create_mesh_fast(self, vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """优化网格创建：用pyvista批量接口"""
        if len(vertices) == 0 or len(faces) == 0:
            return pv.PolyData()
        return pv.PolyData.from_regular_faces(vertices, faces)

    def _create_simplified_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """创建简化网格（大场景快速预览用）"""
        if len(vertices) <= LARGE_SCENE_THRESHOLD:
            # 小场景直接返回完整网格
            return self._create_mesh_fast(vertices, faces)

        # 大场景：每隔4个面元保留1个（减少80%面数）
        simplified_faces = faces[::4]
        return self._create_mesh_fast(vertices, simplified_faces)

    def _group_by_depth(self, nodes: List[bvh_pb.BVHNodeProto], is_aabb: bool) -> Dict[int, List[Dict]]:
        """按深度分组BVH节点（优化循环效率）"""
        depth_dict = {}
        for node in nodes:
            converted_node = self._convert_node_bound(node, is_aabb)
            d = converted_node["depth"]
            if d not in depth_dict:
                depth_dict[d] = []
            depth_dict[d].append(converted_node)
        return depth_dict

    def _convert_node_bound(self, node: bvh_pb.BVHNodeProto, is_aabb: bool) -> Dict:
        """将Protobuf节点转为字典格式"""
        new_node = {
            "node_id": node.node_id,
            "depth": node.depth,
            "node_type": node.node_type,
            "bound_type": node.bound_type,
            "triangle_indices": list(node.triangle_indices)
        }
        if is_aabb:
            aabb = node.aabb_bound
            min_coord = convert_coordinate(np.array([aabb.min.x, aabb.min.y, aabb.min.z]))
            max_coord = convert_coordinate(np.array([aabb.max.x, aabb.max.y, aabb.max.z]))
            new_node["bound"] = {
                "min": {"x": min_coord[0], "y": min_coord[1], "z": min_coord[2]},
                "max": {"x": max_coord[0], "y": max_coord[1], "z": max_coord[2]}
            }
        else:
            sphere = node.sphere_bound
            center_coord = convert_coordinate(np.array([sphere.center.x, sphere.center.y, sphere.center.z]))
            new_node["bound"] = {
                "center": {"x": center_coord[0], "y": center_coord[1], "z": center_coord[2]},
                "radius": sphere.radius
            }
        return new_node

    def _calculate_extent(self, converted_vertices: np.ndarray) -> np.ndarray:
        """计算场景范围"""
        min_bound = np.min(converted_vertices, axis=0)
        max_bound = np.max(converted_vertices, axis=0)
        return max_bound - min_bound

    def _calculate_center(self, converted_vertices: np.ndarray) -> np.ndarray:
        """计算场景中心"""
        min_bound = np.min(converted_vertices, axis=0)
        max_bound = np.max(converted_vertices, axis=0)
        return (min_bound + max_bound) / 2.0


# ====================== 数据解析类（优化后） ======================
class BVHDataLoader:
    def __init__(self, aabb_path: str, sphere_path: str, obj_path: str, ray_path: str):
        self.aabb_path = aabb_path
        self.sphere_path = sphere_path
        self.obj_path = obj_path
        self.ray_path = ray_path

        # 核心数据（惰性初始化）
        self.aabb_bvh = None
        self.sphere_bvh = None
        self.ray_data = None
        self.vertices = None
        self.faces = None
        self.converted_vertices = None
        self.aabb_depth_nodes = None
        self.sphere_depth_nodes = None
        self.full_scene_mesh = None
        self.simplified_mesh = None
        self.scene_extent = np.array([10.0, 10.0, 10.0])
        self.scene_center = np.array([0.0, 0.0, 0.0])

        # 缓存管理器
        self.cache_manager = CacheManager()

    def reload_data(self, async_loader: AsyncDataLoader = None):
        """重新加载数据（支持异步加载结果同步）"""
        if async_loader:
            # 从异步加载结果同步数据
            self.aabb_bvh = async_loader.aabb_bvh
            self.sphere_bvh = async_loader.sphere_bvh
            self.ray_data = async_loader.ray_data
            self.vertices = async_loader.vertices
            self.faces = async_loader.faces
            self.converted_vertices = async_loader.converted_vertices
            self.aabb_depth_nodes = async_loader.aabb_depth_nodes
            self.sphere_depth_nodes = async_loader.sphere_depth_nodes
            self.full_scene_mesh = async_loader.full_scene_mesh
            self.simplified_mesh = async_loader.simplified_mesh
            self.scene_extent = async_loader.scene_extent
            self.scene_center = async_loader.scene_center
        else:
            # 同步加载（小场景 fallback）
            async_loader = AsyncDataLoader(self.aabb_path, self.sphere_path, self.obj_path, self.ray_path)
            async_loader.run()
            self.reload_data(async_loader)

    # 以下接口保持不变（兼容原有UI逻辑）
    def get_nodes_at_depth(self, bvh_type: str, depth: int) -> List[Dict]:
        if bvh_type == "AABB":
            return self.aabb_depth_nodes.get(depth, []) if self.aabb_depth_nodes else []
        else:
            return self.sphere_depth_nodes.get(depth, []) if self.sphere_depth_nodes else []

    def get_node_by_id(self, bvh_type: str, node_id: int) -> Dict:
        all_nodes = []
        if bvh_type == "AABB" and self.aabb_depth_nodes:
            for nodes in self.aabb_depth_nodes.values():
                all_nodes.extend(nodes)
        elif bvh_type == "Sphere" and self.sphere_depth_nodes:
            for nodes in self.sphere_depth_nodes.values():
                all_nodes.extend(nodes)

        for n in all_nodes:
            if n["node_id"] == node_id:
                return n
        raise ValueError(f"节点ID {node_id} 不存在")

    def get_node_triangles_mesh(self, node: Dict) -> pv.PolyData:
        tri_indices = node["triangle_indices"]
        if len(tri_indices) == 0 or self.faces is None:
            return pv.PolyData()
        node_faces = self.faces[tri_indices]
        return pv.PolyData.from_regular_faces(self.vertices, node_faces)

    def get_all_depths(self, bvh_type: str) -> List[int]:
        if bvh_type == "AABB" and self.aabb_depth_nodes:
            return sorted(self.aabb_depth_nodes.keys())
        elif bvh_type == "Sphere" and self.sphere_depth_nodes:
            return sorted(self.sphere_depth_nodes.keys())
        return [0]

    def get_scene_mesh(self, use_simplified: bool = False) -> pv.PolyData:
        """获取场景网格（支持简化版/完整版切换）"""
        if use_simplified and self.simplified_mesh is not None:
            return self.simplified_mesh
        return self.full_scene_mesh if self.full_scene_mesh is not None else pv.PolyData()


# ====================== 工具函数（不变） ======================
def set_widgets_enabled(layout: Union[QVBoxLayout, QHBoxLayout], enabled: bool):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():
            widget = item.widget()
            widget.setEnabled(enabled)
            palette = widget.palette()
            if not enabled:
                palette.setColor(QPalette.Button, QColor(220, 220, 220))
                palette.setColor(QPalette.Window, QColor(240, 240, 240))
                palette.setColor(QPalette.Text, QColor(120, 120, 120))
                palette.setColor(QPalette.ButtonText, QColor(120, 120, 120))
            else:
                palette = QApplication.palette()
            widget.setPalette(palette)
            widget.repaint()
        elif item.layout():
            set_widgets_enabled(item.layout(), enabled)


# ====================== 主窗口类（优化加载逻辑，UI不变） ======================
class BVHMainWindow(QMainWindow):
    def __init__(self, data_loader: BVHDataLoader, aabb_path: str, sphere_path: str, obj_path: str, ray_path: str):
        super().__init__()
        self.data = data_loader
        self.paths = (aabb_path, sphere_path, obj_path, ray_path)
        self.setWindowTitle("BVH可视化")
        self.setGeometry(350, 50, 1920, 1300)

        # 核心状态（不变）
        self.current_bvh = "AABB"
        self.current_depth = 0
        self.current_node_id = 0
        self.show_ray = True
        self.user_show_scene = True
        self.force_show_scene = True
        self.highlight_node_tri = True
        self.show_axes = False
        self.show_bvh_structure = True
        self.first_render = True
        self.use_simplified_mesh = False  # 是否使用简化网格
        self.full_data_loaded = False  # 完整数据是否加载完成

        # 样式参数（不变）
        self.scene_surface_color = "lightgray"
        self.scene_wire_color = "gray"
        self.scene_wire_width = 3.0
        self.scene_opacity = 0.9
        self.scene_ambient = 0.1
        self.aabb_surface_color = "red"
        self.aabb_wire_color = "blue"
        self.aabb_wire_width = 5.0
        self.aabb_opacity = 0.7
        self.aabb_ambient = 0.8
        self.sphere_surface_color = "blue"
        self.sphere_wire_color = "gray"
        self.sphere_wire_width = 1.0
        self.sphere_opacity = 0.5
        self.sphere_ambient = 0.8
        self.tri_surface_color = "yellow"
        self.tri_wire_color = "red"
        self.tri_wire_width = 3.0
        self.tri_opacity = 0.5
        self.tri_ambient = 0.3
        self.ray_main_color = "black"
        self.ray_arrow_color = "purple"
        self.ray_origin_color = "black"
        self.ray_line_width = 6.0
        self.ray_length_factor = 3.0
        self.ray_origin_size_factor = 0.03
        self.ray_arrow_length_factor = 0.6
        self.hit_point_color = "purple"
        self.hit_point_size = 18
        self.axes_line_width = 5.5
        self.axes_total_length = np.max(self.data.scene_extent) * 1.5
        self.axes_arrow_scale = 0.15
        self.axes_label_font_size = 10
        self.axes_arrow_tip_radius_ratio = 0.05
        self.axes_arrow_shaft_radius_ratio = 0.05

        # 初始化UI（不变）
        self._init_ui_layout()
        self._update_style_panel_enabled()
        self._init_3d_renderer()

        # 启动异步加载
        self._start_async_load()

    def _init_ui_layout(self):
        """UI布局初始化（完全不变）"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_control_widget = QWidget()
        top_control_widget.setMinimumHeight(300)
        top_control_widget.setMaximumHeight(400)
        top_layout = QVBoxLayout(top_control_widget)
        top_layout.setContentsMargins(20, 20, 20, 20)
        top_layout.setSpacing(12)

        scroll_content = QWidget()
        main_horizontal_layout = QHBoxLayout(scroll_content)
        main_horizontal_layout.setContentsMargins(15, 15, 15, 15)
        main_horizontal_layout.setSpacing(18)

        three_rows_layout = QVBoxLayout()
        three_rows_layout.setSpacing(12)

        # 第一行：核心控制 + 面元样式
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(18)
        row1_layout.setContentsMargins(0, 0, 0, 0)

        core_group = QGroupBox("核心控制")
        core_group.setStyleSheet(GROUP_BOX_STYLE)
        core_layout = QHBoxLayout(core_group)
        core_layout.setSpacing(14)
        core_layout.setContentsMargins(20, 10, 20, 10)
        core_layout.addWidget(QLabel("包围体类型："))
        self.bvh_combo = QComboBox()
        self.bvh_combo.addItems(["AABB", "Sphere"])
        self.bvh_combo.currentTextChanged.connect(self._on_bvh_change)
        self.bvh_combo.setMinimumWidth(100)
        core_layout.addWidget(self.bvh_combo)
        core_layout.addWidget(QLabel("层级："))
        self.depth_combo = QComboBox()
        self._update_depth_combo()
        self.depth_combo.currentTextChanged.connect(self._on_depth_change)
        self.depth_combo.setMinimumWidth(100)
        core_layout.addWidget(self.depth_combo)
        core_layout.addWidget(QLabel("节点ID："))
        self.node_combo = QComboBox()
        self._update_node_combo()
        self.node_combo.currentTextChanged.connect(self._on_node_change)
        self.node_combo.setMinimumWidth(100)
        core_layout.addWidget(self.node_combo)
        core_layout.addWidget(QLabel("│"))
        self.axes_check = QCheckBox("绘制坐标轴")
        self.axes_check.setChecked(self.show_axes)
        self.axes_check.stateChanged.connect(self._on_axes_toggle)
        self.axes_check.setMinimumWidth(110)
        core_layout.addWidget(self.axes_check)
        self.scene_check = QCheckBox("绘制场景")
        self.scene_check.setChecked(True)
        self.scene_check.stateChanged.connect(self._on_scene_toggle)
        self.scene_check.setMinimumWidth(100)
        core_layout.addWidget(self.scene_check)
        self.tri_check = QCheckBox("绘制所包面元")
        self.tri_check.setChecked(True)
        self.tri_check.stateChanged.connect(self._on_tri_toggle)
        self.tri_check.setMinimumWidth(120)
        core_layout.addWidget(self.tri_check)
        self.ray_check = QCheckBox("绘制射线")
        self.ray_check.setChecked(True)
        self.ray_check.stateChanged.connect(self._on_ray_toggle)
        self.ray_check.setMinimumWidth(100)
        core_layout.addWidget(self.ray_check)
        self.bvh_structure_check = QCheckBox("绘制包围体结构")
        self.bvh_structure_check.setChecked(True)
        self.bvh_structure_check.stateChanged.connect(self._on_bvh_structure_toggle)
        self.bvh_structure_check.setMinimumWidth(130)
        core_layout.addWidget(self.bvh_structure_check)

        tri_group = QGroupBox("面元样式")
        tri_group.setStyleSheet(GROUP_BOX_STYLE)
        tri_layout = QHBoxLayout(tri_group)
        tri_layout.setSpacing(14)
        tri_layout.setContentsMargins(20, 10, 20, 10)
        self.tri_style_layout = tri_layout
        tri_layout.addWidget(QLabel("面元色："))
        self.tri_color_combo = QComboBox()
        self.tri_color_combo.addItems(["yellow", "gold", "orange", "red", "green", "blue", "white", "gray"])
        self.tri_color_combo.setCurrentText(self.tri_surface_color)
        self.tri_color_combo.currentTextChanged.connect(self._on_tri_color_change)
        self.tri_color_combo.setMinimumWidth(100)
        tri_layout.addWidget(self.tri_color_combo)
        tri_layout.addWidget(QLabel("框线色："))
        self.tri_wire_color_combo = QComboBox()
        self.tri_wire_color_combo.addItems(["black", "white", "red", "green", "blue", "gray", "yellow"])
        self.tri_wire_color_combo.setCurrentText(self.tri_wire_color)
        self.tri_wire_color_combo.currentTextChanged.connect(self._on_tri_wire_color_change)
        self.tri_wire_color_combo.setMinimumWidth(100)
        tri_layout.addWidget(self.tri_wire_color_combo)
        tri_layout.addWidget(QLabel("框线宽："))
        self.tri_wire_width_spin = QDoubleSpinBox()
        self.tri_wire_width_spin.setRange(0.1, 5.0)
        self.tri_wire_width_spin.setValue(self.tri_wire_width)
        self.tri_wire_width_spin.setSingleStep(0.1)
        self.tri_wire_width_spin.valueChanged.connect(self._on_tri_wire_width_change)
        self.tri_wire_width_spin.setMinimumWidth(90)
        tri_layout.addWidget(self.tri_wire_width_spin)
        tri_layout.addWidget(QLabel("透明度："))
        self.tri_opacity_spin = QDoubleSpinBox()
        self.tri_opacity_spin.setRange(0.0, 1.0)
        self.tri_opacity_spin.setValue(self.tri_opacity)
        self.tri_opacity_spin.setSingleStep(0.1)
        self.tri_opacity_spin.valueChanged.connect(self._on_tri_opacity_change)
        self.tri_opacity_spin.setMinimumWidth(90)
        tri_layout.addWidget(self.tri_opacity_spin)

        row1_layout.addWidget(core_group)
        row1_layout.addWidget(tri_group)
        row1_layout.addStretch()
        three_rows_layout.addLayout(row1_layout)

        # 第二行：场景样式 + 射线样式 + 重新加载按钮
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(18)
        row2_layout.setContentsMargins(0, 0, 0, 0)

        scene_group = QGroupBox("场景样式")
        scene_group.setStyleSheet(GROUP_BOX_STYLE)
        scene_layout = QHBoxLayout(scene_group)
        scene_layout.setSpacing(14)
        scene_layout.setContentsMargins(20, 10, 20, 10)
        self.scene_style_layout = scene_layout
        scene_layout.addWidget(QLabel("面元色："))
        self.scene_color_combo = QComboBox()
        self.scene_color_combo.addItems(["lightgray", "white", "gray", "black", "red", "green", "blue"])
        self.scene_color_combo.setCurrentText(self.scene_surface_color)
        self.scene_color_combo.currentTextChanged.connect(self._on_scene_color_change)
        self.scene_color_combo.setMinimumWidth(100)
        scene_layout.addWidget(self.scene_color_combo)
        scene_layout.addWidget(QLabel("框线色："))
        self.scene_wire_color_combo = QComboBox()
        self.scene_wire_color_combo.addItems(["gray", "black", "white", "red", "green", "blue"])
        self.scene_wire_color_combo.setCurrentText(self.scene_wire_color)
        self.scene_wire_color_combo.currentTextChanged.connect(self._on_scene_wire_color_change)
        self.scene_wire_color_combo.setMinimumWidth(100)
        scene_layout.addWidget(self.scene_wire_color_combo)
        scene_layout.addWidget(QLabel("框线宽："))
        self.scene_wire_width_spin = QDoubleSpinBox()
        self.scene_wire_width_spin.setRange(0.1, 10.0)
        self.scene_wire_width_spin.setValue(self.scene_wire_width)
        self.scene_wire_width_spin.setSingleStep(0.1)
        self.scene_wire_width_spin.valueChanged.connect(self._on_scene_wire_width_change)
        self.scene_wire_width_spin.setMinimumWidth(90)
        scene_layout.addWidget(self.scene_wire_width_spin)
        scene_layout.addWidget(QLabel("透明度："))
        self.scene_opacity_spin = QDoubleSpinBox()
        self.scene_opacity_spin.setRange(0.0, 1.0)
        self.scene_opacity_spin.setValue(self.scene_opacity)
        self.scene_opacity_spin.setSingleStep(0.1)
        self.scene_opacity_spin.valueChanged.connect(self._on_scene_opacity_change)
        self.scene_opacity_spin.setMinimumWidth(90)
        scene_layout.addWidget(self.scene_opacity_spin)

        ray_group = QGroupBox("射线样式")
        ray_group.setStyleSheet(GROUP_BOX_STYLE)
        ray_layout = QHBoxLayout(ray_group)
        ray_layout.setSpacing(14)
        ray_layout.setContentsMargins(20, 10, 20, 10)
        self.ray_style_layout = ray_layout
        ray_layout.addWidget(QLabel("主体色："))
        self.ray_main_color_combo = QComboBox()
        self.ray_main_color_combo.addItems(["limegreen", "green", "red", "blue", "yellow", "white", "black"])
        self.ray_main_color_combo.setCurrentText(self.ray_main_color)
        self.ray_main_color_combo.currentTextChanged.connect(self._on_ray_main_color_change)
        self.ray_main_color_combo.setMinimumWidth(100)
        ray_layout.addWidget(self.ray_main_color_combo)
        ray_layout.addWidget(QLabel("线宽："))
        self.ray_line_width_spin = QDoubleSpinBox()
        self.ray_line_width_spin.setRange(0.1, 10.0)
        self.ray_line_width_spin.setValue(self.ray_line_width)
        self.ray_line_width_spin.setSingleStep(0.1)
        self.ray_line_width_spin.valueChanged.connect(self._on_ray_line_width_change)
        self.ray_line_width_spin.setMinimumWidth(90)
        ray_layout.addWidget(self.ray_line_width_spin)
        ray_layout.addWidget(QLabel("长度系数："))
        self.ray_length_spin = QDoubleSpinBox()
        self.ray_length_spin.setRange(1.0, 2.0)
        self.ray_length_spin.setValue(self.ray_length_factor)
        self.ray_length_spin.setSingleStep(0.05)
        self.ray_length_spin.valueChanged.connect(self._on_ray_length_change)
        self.ray_length_spin.setMinimumWidth(90)
        ray_layout.addWidget(self.ray_length_spin)
        ray_layout.addWidget(QLabel("碰撞点大小："))
        self.hit_point_size_spin = QSpinBox()
        self.hit_point_size_spin.setRange(5, 30)
        self.hit_point_size_spin.setValue(self.hit_point_size)
        self.hit_point_size_spin.setSingleStep(1)
        self.hit_point_size_spin.valueChanged.connect(self._on_hit_point_size_change)
        self.hit_point_size_spin.setMinimumWidth(90)
        ray_layout.addWidget(self.hit_point_size_spin)

        reload_group = QGroupBox("文件操作")
        reload_group.setStyleSheet(GROUP_BOX_STYLE)
        reload_layout = QHBoxLayout(reload_group)
        reload_layout.setContentsMargins(20, 10, 20, 10)
        reload_layout.setSpacing(0)
        self.reload_btn = QPushButton("重新加载最新文件", self)
        self.reload_btn.setStyleSheet(GROUP_BOX_STYLE.replace("QGroupBox", "QPushButton"))
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        self.reload_btn.setMinimumWidth(160)
        reload_layout.addWidget(self.reload_btn, alignment=Qt.AlignCenter)

        row2_layout.addWidget(scene_group)
        row2_layout.addWidget(ray_group)
        row2_layout.addWidget(reload_group)
        three_rows_layout.addLayout(row2_layout)

        # 第三行：AABB盒样式 + 包围球样式
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(18)
        row3_layout.setContentsMargins(0, 0, 0, 0)

        aabb_group = QGroupBox("AABB盒样式")
        aabb_group.setStyleSheet(GROUP_BOX_STYLE)
        aabb_layout = QHBoxLayout(aabb_group)
        aabb_layout.setSpacing(14)
        aabb_layout.setContentsMargins(20, 10, 20, 10)
        self.aabb_style_layout = aabb_layout
        aabb_layout.addWidget(QLabel("盒面色："))
        self.aabb_color_combo = QComboBox()
        self.aabb_color_combo.addItems(["red", "green", "blue", "yellow", "orange", "purple", "black", "white"])
        self.aabb_color_combo.setCurrentText(self.aabb_surface_color)
        self.aabb_color_combo.currentTextChanged.connect(self._on_aabb_color_change)
        self.aabb_color_combo.setMinimumWidth(100)
        aabb_layout.addWidget(self.aabb_color_combo)
        aabb_layout.addWidget(QLabel("框线色："))
        self.aabb_wire_color_combo = QComboBox()
        self.aabb_wire_color_combo.addItems(["black", "white", "red", "green", "blue", "gray"])
        self.aabb_wire_color_combo.setCurrentText(self.aabb_wire_color)
        self.aabb_wire_color_combo.currentTextChanged.connect(self._on_aabb_wire_color_change)
        self.aabb_wire_color_combo.setMinimumWidth(100)
        aabb_layout.addWidget(self.aabb_wire_color_combo)
        aabb_layout.addWidget(QLabel("框线宽："))
        self.aabb_wire_width_spin = QSpinBox()
        self.aabb_wire_width_spin.setRange(1, 20)
        self.aabb_wire_width_spin.setValue(int(self.aabb_wire_width))
        self.aabb_wire_width_spin.setSingleStep(1)
        self.aabb_wire_width_spin.valueChanged.connect(self._on_aabb_wire_width_change)
        self.aabb_wire_width_spin.setMinimumWidth(90)
        aabb_layout.addWidget(self.aabb_wire_width_spin)
        aabb_layout.addWidget(QLabel("透明度："))
        self.aabb_opacity_spin = QDoubleSpinBox()
        self.aabb_opacity_spin.setRange(0.0, 1.0)
        self.aabb_opacity_spin.setValue(self.aabb_opacity)
        self.aabb_opacity_spin.setSingleStep(0.1)
        self.aabb_opacity_spin.valueChanged.connect(self._on_aabb_opacity_change)
        self.aabb_opacity_spin.setMinimumWidth(90)
        aabb_layout.addWidget(self.aabb_opacity_spin)

        sphere_group = QGroupBox("包围球样式")
        sphere_group.setStyleSheet(GROUP_BOX_STYLE)
        sphere_layout = QHBoxLayout(sphere_group)
        sphere_layout.setSpacing(14)
        sphere_layout.setContentsMargins(20, 10, 20, 10)
        self.sphere_style_layout = sphere_layout
        sphere_layout.addWidget(QLabel("球面色："))
        self.sphere_color_combo = QComboBox()
        self.sphere_color_combo.addItems(["blue", "red", "green", "yellow", "orange", "purple", "black", "white"])
        self.sphere_color_combo.setCurrentText(self.sphere_surface_color)
        self.sphere_color_combo.currentTextChanged.connect(self._on_sphere_color_change)
        self.sphere_color_combo.setMinimumWidth(100)
        sphere_layout.addWidget(self.sphere_color_combo)
        sphere_layout.addWidget(QLabel("框线色："))
        self.sphere_wire_color_combo = QComboBox()
        self.sphere_wire_color_combo.addItems(["black", "white", "red", "green", "blue", "gray"])
        self.sphere_wire_color_combo.setCurrentText(self.sphere_wire_color)
        self.sphere_wire_color_combo.currentTextChanged.connect(self._on_sphere_wire_color_change)
        self.sphere_wire_color_combo.setMinimumWidth(100)
        sphere_layout.addWidget(self.sphere_wire_color_combo)
        sphere_layout.addWidget(QLabel("框线宽："))
        self.sphere_wire_width_spin = QSpinBox()
        self.sphere_wire_width_spin.setRange(1, 20)
        self.sphere_wire_width_spin.setValue(int(self.sphere_wire_width))
        self.sphere_wire_width_spin.setSingleStep(1)
        self.sphere_wire_width_spin.valueChanged.connect(self._on_sphere_wire_width_change)
        self.sphere_wire_width_spin.setMinimumWidth(90)
        sphere_layout.addWidget(self.sphere_wire_width_spin)
        sphere_layout.addWidget(QLabel("透明度："))
        self.sphere_opacity_spin = QDoubleSpinBox()
        self.sphere_opacity_spin.setRange(0.0, 1.0)
        self.sphere_opacity_spin.setValue(self.sphere_opacity)
        self.sphere_opacity_spin.setSingleStep(0.1)
        self.sphere_opacity_spin.valueChanged.connect(self._on_sphere_opacity_change)
        self.sphere_opacity_spin.setMinimumWidth(90)
        sphere_layout.addWidget(self.sphere_opacity_spin)

        row3_layout.addWidget(aabb_group)
        row3_layout.addWidget(sphere_group)
        row3_layout.addStretch()
        three_rows_layout.addLayout(row3_layout)

        main_horizontal_layout.addLayout(three_rows_layout)
        top_layout.addWidget(scroll_content)
        main_layout.addWidget(top_control_widget)

        self.render_widget = QtInteractor(main_widget)
        main_layout.addWidget(self.render_widget, stretch=1)

    def _init_3d_renderer(self):
        """3D渲染器初始化（不变）"""
        self.render_widget.enable_anti_aliasing()
        self.render_widget.background_color = "white"
        self.render_widget.renderer.use_depth_peeling = True
        self.render_widget.renderer.max_number_of_peels = 20
        self.render_widget.renderer.opacity_unit_distance = 0.1
        self.render_widget.renderer.reset_camera()

    def _start_async_load(self):
        """启动异步加载并显示进度条"""
        aabb_path, sphere_path, obj_path, ray_path = self.paths
        self.async_loader = AsyncDataLoader(aabb_path, sphere_path, obj_path, ray_path)

        # 创建加载进度条
        self.progress_dialog = QProgressDialog("正在加载数据...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("加载进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)

        # 连接信号
        self.async_loader.progress_update.connect(self._on_load_progress)
        self.async_loader.load_finished.connect(self._on_load_finished)
        self.async_loader.load_failed.connect(self._on_load_failed)
        self.progress_dialog.canceled.connect(self.async_loader.quit)

        # 启动线程
        self.async_loader.start()

    @pyqtSlot(str, int)
    def _on_load_progress(self, desc: str, percent: int):
        """更新加载进度"""
        self.progress_dialog.setLabelText(desc)
        self.progress_dialog.setValue(percent)

    @pyqtSlot(dict)
    def _on_load_finished(self, data: Dict):
        """异步加载完成，更新数据并渲染"""
        self.progress_dialog.close()

        # 同步数据到DataLoader
        self.data.aabb_bvh = data["aabb_bvh"]
        self.data.sphere_bvh = data["sphere_bvh"]
        self.data.ray_data = data["ray_data"]
        self.data.vertices = data["vertices"]
        self.data.faces = data["faces"]
        self.data.converted_vertices = data["converted_vertices"]
        self.data.aabb_depth_nodes = data["aabb_depth_nodes"]
        self.data.sphere_depth_nodes = data["sphere_depth_nodes"]
        self.data.full_scene_mesh = data["full_scene_mesh"]
        self.data.simplified_mesh = data["simplified_mesh"]
        self.data.scene_extent = data["scene_extent"]
        self.data.scene_center = data["scene_center"]

        # 更新状态
        self.full_data_loaded = True
        self.axes_total_length = np.max(self.data.scene_extent) * 1.5

        # 更新UI选项
        self._update_depth_combo()
        self._update_node_combo()

        # 首次渲染（大场景先显示简化网格）
        self.use_simplified_mesh = len(self.data.vertices) > LARGE_SCENE_THRESHOLD
        self._update_render()

        print(f"✅ 加载完成：顶点数={len(self.data.vertices)}, 三角形数={len(self.data.faces)}")

    @pyqtSlot(str)
    def _on_load_failed(self, error: str):
        """加载失败处理"""
        self.progress_dialog.close()
        print(f"❌ {error}")
        # 尝试同步加载（降级方案）
        QApplication.processEvents()
        self.progress_dialog = QProgressDialog("加载失败，尝试降级加载...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("降级加载")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        try:
            self.data.reload_data()
            self.full_data_loaded = True
            self.axes_total_length = np.max(self.data.scene_extent) * 1.5
            self._update_depth_combo()
            self._update_node_combo()
            self._update_render()
            self.progress_dialog.close()
            print("✅ 降级加载成功")
        except Exception as e:
            self.progress_dialog.close()
            print(f"❌ 降级加载也失败：{e}")

    # 以下所有事件回调和渲染方法均保持不变
    def _update_depth_combo(self):
        self.depth_combo.clear()
        all_depths = self.data.get_all_depths(self.current_bvh)
        if not all_depths:
            all_depths = [0]
        self.depth_combo.addItems([str(d) for d in all_depths])
        if str(self.current_depth) in [self.depth_combo.itemText(i) for i in range(self.depth_combo.count())]:
            self.depth_combo.setCurrentText(str(self.current_depth))

    def _update_node_combo(self):
        self.node_combo.clear()
        nodes = self.data.get_nodes_at_depth(self.current_bvh, self.current_depth)
        node_ids = [str(n["node_id"]) for n in nodes] if nodes else ["0"]
        self.node_combo.addItems(node_ids)
        if node_ids:
            self.current_node_id = int(node_ids[0])

    def _update_style_panel_enabled(self):
        set_widgets_enabled(self.scene_style_layout, self.scene_check.isChecked())
        set_widgets_enabled(self.tri_style_layout, self.tri_check.isChecked())
        set_widgets_enabled(self.ray_style_layout, self.ray_check.isChecked())
        set_widgets_enabled(self.aabb_style_layout, self.bvh_structure_check.isChecked())
        set_widgets_enabled(self.sphere_style_layout, self.bvh_structure_check.isChecked())

    @pyqtSlot()
    def _on_reload_clicked(self):
        try:
            self.progress_dialog = QProgressDialog("重新加载数据...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("重新加载")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.show()

            # 重启异步加载
            self.async_loader = AsyncDataLoader(*self.paths)
            self.async_loader.progress_update.connect(self._on_load_progress)
            self.async_loader.load_finished.connect(self._on_load_finished)
            self.async_loader.load_failed.connect(self._on_load_failed)
            self.progress_dialog.canceled.connect(self.async_loader.quit)
            self.async_loader.start()

            self.first_render = True
        except Exception as e:
            self.progress_dialog.close()
            print(f"❌ 重新加载失败：{e}")

    @pyqtSlot(int)
    def _on_axes_toggle(self, state: int):
        self.show_axes = (state == Qt.Checked)
        self._update_render()

    @pyqtSlot(int)
    def _on_bvh_structure_toggle(self, state: int):
        self.show_bvh_structure = (state == Qt.Checked)
        self._update_style_panel_enabled()
        self._update_render()

    @pyqtSlot(str)
    def _on_bvh_change(self, bvh_type: str):
        self.current_bvh = bvh_type
        self._update_depth_combo()
        self._update_node_combo()
        self._update_render()

    @pyqtSlot(str)
    def _on_depth_change(self, depth_str: str):
        try:
            self.current_depth = int(depth_str)
            self._update_node_combo()
            self._update_render()
        except ValueError:
            print(f"无效的层级值：{depth_str}")

    @pyqtSlot(str)
    def _on_node_change(self, node_id: str):
        if node_id.isdigit():
            self.current_node_id = int(node_id)
            self._update_render()

    @pyqtSlot(int)
    def _on_scene_toggle(self, state: int):
        self.user_show_scene = (state == Qt.Checked)
        self._update_style_panel_enabled()
        self._update_render()

    @pyqtSlot(int)
    def _on_tri_toggle(self, state: int):
        self.highlight_node_tri = (state == Qt.Checked)
        self._update_style_panel_enabled()
        self._update_render()

    @pyqtSlot(int)
    def _on_ray_toggle(self, state: int):
        self.show_ray = (state == Qt.Checked)
        self._update_style_panel_enabled()
        self._update_render()

    @pyqtSlot(str)
    def _on_scene_color_change(self, color: str):
        self.scene_surface_color = color
        self._update_render()

    @pyqtSlot(str)
    def _on_scene_wire_color_change(self, color: str):
        self.scene_wire_color = color
        self._update_render()

    @pyqtSlot(float)
    def _on_scene_wire_width_change(self, width: float):
        self.scene_wire_width = width
        self._update_render()

    @pyqtSlot(float)
    def _on_scene_opacity_change(self, opacity: float):
        self.scene_opacity = opacity
        self._update_render()

    @pyqtSlot(str)
    def _on_aabb_color_change(self, color: str):
        self.aabb_surface_color = color
        self._update_render()

    @pyqtSlot(str)
    def _on_aabb_wire_color_change(self, color: str):
        self.aabb_wire_color = color
        self._update_render()

    @pyqtSlot(int)
    def _on_aabb_wire_width_change(self, width: int):
        self.aabb_wire_width = width
        self._update_render()

    @pyqtSlot(float)
    def _on_aabb_opacity_change(self, opacity: float):
        self.aabb_opacity = opacity
        self._update_render()

    @pyqtSlot(str)
    def _on_sphere_color_change(self, color: str):
        self.sphere_surface_color = color
        self._update_render()

    @pyqtSlot(str)
    def _on_sphere_wire_color_change(self, color: str):
        self.sphere_wire_color = color
        self._update_render()

    @pyqtSlot(int)
    def _on_sphere_wire_width_change(self, width: int):
        self.sphere_wire_width = width
        self._update_render()

    @pyqtSlot(float)
    def _on_sphere_opacity_change(self, opacity: float):
        self.sphere_opacity = opacity
        self._update_render()

    @pyqtSlot(str)
    def _on_tri_color_change(self, color: str):
        self.tri_surface_color = color
        self._update_render()

    @pyqtSlot(str)
    def _on_tri_wire_color_change(self, color: str):
        self.tri_wire_color = color
        self._update_render()

    @pyqtSlot(float)
    def _on_tri_wire_width_change(self, width: float):
        self.tri_wire_width = width
        self._update_render()

    @pyqtSlot(float)
    def _on_tri_opacity_change(self, opacity: float):
        self.tri_opacity = opacity
        self._update_render()

    @pyqtSlot(str)
    def _on_ray_main_color_change(self, color: str):
        self.ray_main_color = color
        self._update_render()

    @pyqtSlot(float)
    def _on_ray_line_width_change(self, width: float):
        self.ray_line_width = width
        self._update_render()

    @pyqtSlot(float)
    def _on_ray_length_change(self, factor: float):
        self.ray_length_factor = factor
        self._update_render()

    @pyqtSlot(int)
    def _on_hit_point_size_change(self, size: int):
        self.hit_point_size = size
        self._update_render()

    def _update_render(self):
        """渲染方法（仅修改场景网格加载逻辑）"""
        self.render_widget.clear()
        is_switching = False
        if hasattr(self, '_last_depth') and hasattr(self, '_last_node_id'):
            if self.current_depth != self._last_depth or self.current_node_id != self._last_node_id:
                is_switching = True
        else:
            is_switching = True
        self._last_depth = self.current_depth
        self._last_node_id = self.current_node_id
        current_show_scene = self.force_show_scene if is_switching else self.user_show_scene

        # 加载场景网格（支持简化版/完整版）
        if current_show_scene and self.full_data_loaded:
            scene_mesh = self.data.get_scene_mesh(use_simplified=self.use_simplified_mesh)
            if scene_mesh.n_points > 0:
                self.render_widget.add_mesh(
                    scene_mesh,
                    color=self.scene_surface_color,
                    opacity=self.scene_opacity,
                    style="surface",
                    ambient=self.scene_ambient,
                    reset_camera=False
                )
                self.render_widget.add_mesh(
                    scene_mesh,
                    color=self.scene_wire_color,
                    style="wireframe",
                    line_width=self.scene_wire_width,
                    ambient=self.scene_ambient,
                    reset_camera=False
                )

        # 以下渲染逻辑均不变
        if self.show_axes and self.full_data_loaded:
            axes_origin = convert_coordinate(self.data.scene_center)
            ray_total_length = self.axes_total_length
            arrow_length = ray_total_length * self.axes_arrow_scale
            tip_radius = arrow_length * self.axes_arrow_tip_radius_ratio
            shaft_radius = arrow_length * self.axes_arrow_shaft_radius_ratio

            x_dir = np.array([1.0, 0.0, 0.0])
            x_ray_end = axes_origin + x_dir * ray_total_length
            x_ray = pv.Line(axes_origin, x_ray_end)
            self.render_widget.add_mesh(
                x_ray, color="red", line_width=self.axes_line_width, style="wireframe", reset_camera=False
            )
            x_arrow_start = x_ray_end
            x_arrow = pv.Arrow(start=x_arrow_start, direction=x_dir, tip_length=arrow_length, tip_radius=tip_radius,
                               shaft_radius=shaft_radius)
            self.render_widget.add_mesh(x_arrow, color="red", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("X", x_ray_end + x_dir * (arrow_length * 0.5),
                                        font_size=self.axes_label_font_size, color="red", shadow=True)

            y_dir = np.array([0.0, 1.0, 0.0])
            y_ray_end = axes_origin + y_dir * ray_total_length
            y_ray = pv.Line(axes_origin, y_ray_end)
            self.render_widget.add_mesh(
                y_ray, color="green", line_width=self.axes_line_width, style="wireframe", reset_camera=False
            )
            y_arrow_start = y_ray_end
            y_arrow = pv.Arrow(start=y_arrow_start, direction=y_dir, tip_length=arrow_length, tip_radius=tip_radius,
                               shaft_radius=shaft_radius)
            self.render_widget.add_mesh(y_arrow, color="green", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("Y", y_ray_end + y_dir * (arrow_length * 0.5),
                                        font_size=self.axes_label_font_size, color="green", shadow=True)

            z_dir = np.array([0.0, 0.0, 1.0])
            z_ray_end = axes_origin + z_dir * ray_total_length
            z_ray = pv.Line(axes_origin, z_ray_end)
            self.render_widget.add_mesh(
                z_ray, color="blue", line_width=self.axes_line_width, style="wireframe", reset_camera=False
            )
            z_arrow_start = z_ray_end
            z_arrow = pv.Arrow(start=z_arrow_start, direction=z_dir, tip_length=arrow_length, tip_radius=tip_radius,
                               shaft_radius=shaft_radius)
            self.render_widget.add_mesh(z_arrow, color="blue", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("Z", z_ray_end + z_dir * (arrow_length * 0.5),
                                        font_size=self.axes_label_font_size, color="blue", shadow=True)

        try:
            if self.show_bvh_structure and self.full_data_loaded:
                node = self.data.get_node_by_id(self.current_bvh, self.current_node_id)
                if self.current_bvh == "AABB":
                    min_p = np.array([node["bound"]["min"]["x"], node["bound"]["min"]["y"], node["bound"]["min"]["z"]])
                    max_p = np.array([node["bound"]["max"]["x"], node["bound"]["max"]["y"], node["bound"]["max"]["z"]])
                    bvh_mesh = pv.Box(bounds=(min_p[0], max_p[0], min_p[1], max_p[1], min_p[2], max_p[2]))
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.aabb_surface_color, opacity=self.aabb_opacity,
                        style="surface", ambient=self.aabb_ambient, reset_camera=False
                    )
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.aabb_wire_color, style="wireframe",
                        line_width=self.aabb_wire_width, ambient=0.5, reset_camera=False
                    )
                else:
                    center = np.array(
                        [node["bound"]["center"]["x"], node["bound"]["center"]["y"], node["bound"]["center"]["z"]])
                    bvh_mesh = pv.Sphere(center=center, radius=node["bound"]["radius"], theta_resolution=30,
                                         phi_resolution=30)
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.sphere_surface_color, opacity=self.sphere_opacity,
                        style="surface", ambient=self.sphere_ambient, reset_camera=False
                    )
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.sphere_wire_color, style="wireframe",
                        line_width=self.sphere_wire_width, ambient=0.5, reset_camera=False
                    )

            if self.highlight_node_tri and self.full_data_loaded:
                node = self.data.get_node_by_id(self.current_bvh, self.current_node_id)
                tri_mesh = self.data.get_node_triangles_mesh(node)
                if tri_mesh.n_points > 0:
                    self.render_widget.add_mesh(
                        tri_mesh, color=self.tri_surface_color, opacity=self.tri_opacity,
                        style="surface", ambient=self.tri_ambient, reset_camera=False
                    )
                    self.render_widget.add_mesh(
                        tri_mesh, color=self.tri_wire_color, style="wireframe",
                        line_width=self.tri_wire_width, ambient=0.5, reset_camera=False
                    )

            if self.show_ray and self.full_data_loaded and hasattr(self.data.ray_data, "ray"):
                ray = self.data.ray_data.ray
                origin = convert_coordinate(np.array([ray.origin.x, ray.origin.y, ray.origin.z]))
                direction = convert_coordinate(np.array([ray.direction.x, ray.direction.y, ray.direction.z]))
                max_dist = 10.0
                if self.data.ray_data.collisions:
                    max_dist = max([hit.distance for hit in self.data.ray_data.collisions])
                extra_length = max_dist * (self.ray_length_factor - 1.0)
                end = origin + direction * (max_dist + extra_length)

                origin_mesh = pv.Sphere(center=origin, radius=max_dist * self.ray_origin_size_factor)
                self.render_widget.add_mesh(
                    origin_mesh, color=self.ray_origin_color, style="surface",
                    ambient=0.6, opacity=0.8, reset_camera=False
                )
                self.render_widget.add_mesh(
                    pv.Line(origin, end), color=self.ray_main_color,
                    line_width=self.ray_line_width, style="wireframe", reset_camera=False
                )
                arrow_start = end
                arrow_mesh = pv.Arrow(
                    start=arrow_start, direction=direction,
                    tip_length=(max_dist + extra_length) * self.ray_arrow_length_factor
                )
                self.render_widget.add_mesh(
                    arrow_mesh, color=self.ray_arrow_color, style="surface",
                    ambient=0.6, opacity=0.8, reset_camera=False
                )

                hit_pts = []
                for hit in self.data.ray_data.collisions:
                    hit_pt = convert_coordinate(np.array([hit.hit_point.x, hit.hit_point.y, hit.hit_point.z]))
                    hit_pts.append(hit_pt)
                if hit_pts:
                    hit_mesh = pv.PolyData(np.array(hit_pts))
                    self.render_widget.add_mesh(
                        hit_mesh, color=self.hit_point_color, point_size=self.hit_point_size,
                        render_points_as_spheres=True, ambient=0.5, reset_camera=False
                    )
        except Exception as e:
            print(f"渲染提示：{str(e)[:100]}")

        if self.first_render and self.full_data_loaded:
            scene_mesh = self.data.get_scene_mesh(use_simplified=self.use_simplified_mesh)
            if scene_mesh.n_points > 0:
                self.render_widget.reset_camera()
            self.first_render = False

        self.render_widget.update()


# ====================== 主函数（不变） ======================
if __name__ == "__main__":
    # 配置文件路径（根据实际项目调整）
    AABB_BVH_PATH = "Cache/m/aabb_bvh_structure.pb"
    SPHERE_BVH_PATH = "Cache/m/sphere_bvh_structure.pb"
    OBJ_PATH = "obj/m.obj"
    RAY_PATH = "Cache/m/ray_hit_data.pb"

    # 检查文件存在性
    for fp in [AABB_BVH_PATH, SPHERE_BVH_PATH, OBJ_PATH, RAY_PATH]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"文件不存在：{fp} → 请检查路径！")

    # 初始化数据加载器和应用
    data_loader = BVHDataLoader(AABB_BVH_PATH, SPHERE_BVH_PATH, OBJ_PATH, RAY_PATH)
    app = QApplication(sys.argv)
    window = BVHMainWindow(data_loader, AABB_BVH_PATH, SPHERE_BVH_PATH, OBJ_PATH, RAY_PATH)
    window.show()
    sys.exit(app.exec_())