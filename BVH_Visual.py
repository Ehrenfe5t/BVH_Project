import sys
import numpy as np
import os
import warnings
from typing import Dict, List, Tuple, Union
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QGroupBox, QSpinBox,
    QDoubleSpinBox, QScrollArea, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPalette, QColor
import pyvista as pv
from pyvistaqt import QtInteractor
# 1. 新增：导入Protobuf生成的模块（需与脚本同级目录）
import BVHData_pb2 as bvh_pb

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# ====================== 数据解析类（核心修改：JSON→Protobuf） ======================
class BVHDataLoader:
    def __init__(self, aabb_path: str, sphere_path: str, obj_path: str, ray_path: str):
        self.aabb_path = aabb_path
        self.sphere_path = sphere_path
        self.obj_path = obj_path
        self.ray_path = ray_path
        self.reload_data()  # 初始化加载

    def reload_data(self):
        """重新加载所有外部文件（核心：读取Protobuf/OBJ）"""
        # 2. 修改：加载Protobuf格式的BVH和射线数据
        self.aabb_bvh = self._load_proto(self.aabb_path, bvh_pb.BVHStructureProto())
        self.sphere_bvh = self._load_proto(self.sphere_path, bvh_pb.BVHStructureProto())
        self.ray_data = self._load_proto(self.ray_path, bvh_pb.RayHitDataProto())
        # OBJ加载逻辑不变
        self.vertices, self.faces = self._load_obj(self.obj_path)
        # 后续可视化依赖的衍生数据（不变）
        self.full_scene_mesh = self._create_full_scene_mesh()
        self.aabb_depth_nodes = self._group_by_depth(self.aabb_bvh.nodes, is_aabb=True)
        self.sphere_depth_nodes = self._group_by_depth(self.sphere_bvh.nodes, is_aabb=False)
        self.scene_origin = np.array([0.0, 0.0, 0.0])
        self.scene_extent = self._calculate_scene_extent()
        self.scene_center = self._calculate_scene_center()

    def _load_proto(self, path: str, proto_msg) -> object:
        """新增：加载Protobuf文件（通用方法，适配所有消息类型）"""
        with open(path, "rb") as f:
            proto_msg.ParseFromString(f.read())  # 二进制解析Protobuf
        return proto_msg

    def _load_obj(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """OBJ加载逻辑不变"""
        vertices, faces = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    faces.append([int(x.split("/")[0]) - 1 for x in parts[1:4]])
        return np.array(vertices, np.float32), np.array(faces, np.int32)

    # 以下方法（_create_full_scene_mesh、_calculate_scene_extent等）均不变，仅适配Protobuf节点格式
    def _create_full_scene_mesh(self) -> pv.PolyData:
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return pv.PolyData()
        faces_flat = np.hstack([np.array([3] * len(self.faces))[:, None], self.faces]).flatten()
        return pv.PolyData(self.vertices, faces_flat)

    def _calculate_scene_extent(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.array([10.0, 10.0, 10.0])
        converted_vertices = np.array([convert_coordinate(v) for v in self.vertices])
        min_bound = np.min(converted_vertices, axis=0)
        max_bound = np.max(converted_vertices, axis=0)
        return max_bound - min_bound

    def _calculate_scene_center(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.array([0.0, 0.0, 0.0])
        converted_vertices = np.array([convert_coordinate(v) for v in self.vertices])
        min_bound = np.min(converted_vertices, axis=0)
        max_bound = np.max(converted_vertices, axis=0)
        return (min_bound + max_bound) / 2.0

    def _convert_node_bound(self, node: bvh_pb.BVHNodeProto, is_aabb: bool) -> Dict:
        """新增：将Protobuf节点转为原有字典格式（兼容下游可视化逻辑）"""
        new_node = {
            "node_id": node.node_id,
            "depth": node.depth,  # Protobuf getter方法，直接调用（无需括号）
            "node_type": node.node_type,
            "bound_type": node.bound_type,
            "triangle_indices": list(node.triangle_indices)  # Protobuf repeated转列表
        }
        # 适配AABB/包围球的Protobuf结构
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

    def _group_by_depth(self, nodes: List[bvh_pb.BVHNodeProto], is_aabb: bool) -> Dict[int, List[Dict]]:
        """适配Protobuf节点列表，按深度分组（逻辑不变）"""
        depth_dict = {}
        for node in nodes:
            converted_node = self._convert_node_bound(node, is_aabb)
            d = converted_node["depth"]
            depth_dict[d] = depth_dict.get(d, []) + [converted_node]
        return depth_dict

    # 以下接口（get_nodes_at_depth、get_node_by_id等）均不变
    def get_nodes_at_depth(self, bvh_type: str, depth: int) -> List[Dict]:
        return self.aabb_depth_nodes.get(depth, []) if bvh_type == "AABB" else self.sphere_depth_nodes.get(depth, [])

    def get_node_by_id(self, bvh_type: str, node_id: int) -> Dict:
        all_nodes = []
        if bvh_type == "AABB":
            for nodes in self.aabb_depth_nodes.values():
                all_nodes.extend(nodes)
        else:
            for nodes in self.sphere_depth_nodes.values():
                all_nodes.extend(nodes)
        for n in all_nodes:
            if n["node_id"] == node_id:
                return n
        raise ValueError(f"节点ID {node_id} 不存在")

    def get_node_triangles_mesh(self, node: Dict) -> pv.PolyData:
        tri_indices = node["triangle_indices"]
        if len(tri_indices) == 0:
            return pv.PolyData()
        node_faces = self.faces[tri_indices]
        faces_flat = np.hstack([np.array([3] * len(node_faces))[:, None], node_faces]).flatten()
        return pv.PolyData(self.vertices, faces_flat)

    def get_all_depths(self, bvh_type: str) -> List[int]:
        if bvh_type == "AABB":
            return sorted(self.aabb_depth_nodes.keys())
        else:
            return sorted(self.sphere_depth_nodes.keys())

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

# ====================== 主窗口类（完全不变，复用可视化逻辑） ======================
class BVHMainWindow(QMainWindow):
    def __init__(self, data_loader: BVHDataLoader):
        super().__init__()
        self.data = data_loader
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
        # 初始化逻辑（不变）
        self._init_ui_layout()
        self._update_style_panel_enabled()
        self._init_3d_renderer()
        self._update_render()

    # 以下所有UI初始化、事件回调、渲染方法均完全不变（省略重复代码，与原逻辑一致）
    def _init_ui_layout(self):
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

        # 第二行：场景样式 + 射线样式 + 重新加载按钮（不变）
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

        # 第三行：AABB盒样式 + 包围球样式（不变）
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

    # 以下所有事件回调（_on_bvh_change、_on_depth_change等）和渲染方法（_update_render）均不变
    def _update_depth_combo(self):
        self.depth_combo.clear()
        all_depths = self.data.get_all_depths(self.current_bvh)
        if not all_depths:
            all_depths = [0]
        self.depth_combo.addItems([str(d) for d in all_depths])
        if str(self.current_depth) in [self.depth_combo.itemText(i) for i in range(self.depth_combo.count())]:
            self.depth_combo.setCurrentText(str(self.current_depth))

    def _update_style_panel_enabled(self):
        set_widgets_enabled(self.scene_style_layout, self.scene_check.isChecked())
        set_widgets_enabled(self.tri_style_layout, self.tri_check.isChecked())
        set_widgets_enabled(self.ray_style_layout, self.ray_check.isChecked())
        set_widgets_enabled(self.aabb_style_layout, self.bvh_structure_check.isChecked())
        set_widgets_enabled(self.sphere_style_layout, self.bvh_structure_check.isChecked())

    def _init_3d_renderer(self):
        self.render_widget.enable_anti_aliasing()
        self.render_widget.background_color = "white"
        self.render_widget.renderer.use_depth_peeling = True
        self.render_widget.renderer.max_number_of_peels = 20
        self.render_widget.renderer.opacity_unit_distance = 0.1
        self.render_widget.renderer.reset_camera()

    def _update_node_combo(self):
        self.node_combo.clear()
        nodes = self.data.get_nodes_at_depth(self.current_bvh, self.current_depth)
        node_ids = [str(n["node_id"]) for n in nodes] if nodes else ["0"]
        self.node_combo.addItems(node_ids)
        if node_ids:
            self.current_node_id = int(node_ids[0])

    @pyqtSlot()
    def _on_reload_clicked(self):
        try:
            self.data.reload_data()
            self._update_depth_combo()
            self._update_node_combo()
            self.first_render = True
            self._update_render()
            print("✅ 成功加载最新文件并重新渲染")
        except Exception as e:
            print(f"❌ 加载文件失败：{str(e)}")

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

        if current_show_scene and self.data.full_scene_mesh.n_points > 0:
            self.render_widget.add_mesh(
                self.data.full_scene_mesh,
                color=self.scene_surface_color,
                opacity=self.scene_opacity,
                style="surface",
                ambient=self.scene_ambient,
                reset_camera=False
            )
            self.render_widget.add_mesh(
                self.data.full_scene_mesh,
                color=self.scene_wire_color,
                style="wireframe",
                line_width=self.scene_wire_width,
                ambient=self.scene_ambient,
                reset_camera=False
            )

        if self.show_axes:
            axes_origin = convert_coordinate(self.data.scene_origin)
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
            x_arrow = pv.Arrow(start=x_arrow_start, direction=x_dir, tip_length=arrow_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
            self.render_widget.add_mesh(x_arrow, color="red", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("X", x_ray_end + x_dir * (arrow_length * 0.5), font_size=self.axes_label_font_size, color="red", shadow=True)

            y_dir = np.array([0.0, 1.0, 0.0])
            y_ray_end = axes_origin + y_dir * ray_total_length
            y_ray = pv.Line(axes_origin, y_ray_end)
            self.render_widget.add_mesh(
                y_ray, color="green", line_width=self.axes_line_width, style="wireframe", reset_camera=False
            )
            y_arrow_start = y_ray_end
            y_arrow = pv.Arrow(start=y_arrow_start, direction=y_dir, tip_length=arrow_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
            self.render_widget.add_mesh(y_arrow, color="green", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("Y", y_ray_end + y_dir * (arrow_length * 0.5), font_size=self.axes_label_font_size, color="green", shadow=True)

            z_dir = np.array([0.0, 0.0, 1.0])
            z_ray_end = axes_origin + z_dir * ray_total_length
            z_ray = pv.Line(axes_origin, z_ray_end)
            self.render_widget.add_mesh(
                z_ray, color="blue", line_width=self.axes_line_width, style="wireframe", reset_camera=False
            )
            z_arrow_start = z_ray_end
            z_arrow = pv.Arrow(start=z_arrow_start, direction=z_dir, tip_length=arrow_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
            self.render_widget.add_mesh(z_arrow, color="blue", style="surface", ambient=0.8, reset_camera=False)
            self.render_widget.add_text("Z", z_ray_end + z_dir * (arrow_length * 0.5), font_size=self.axes_label_font_size, color="blue", shadow=True)

        try:
            if self.show_bvh_structure:
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
                    center = np.array([node["bound"]["center"]["x"], node["bound"]["center"]["y"], node["bound"]["center"]["z"]])
                    bvh_mesh = pv.Sphere(center=center, radius=node["bound"]["radius"], theta_resolution=30, phi_resolution=30)
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.sphere_surface_color, opacity=self.sphere_opacity,
                        style="surface", ambient=self.sphere_ambient, reset_camera=False
                    )
                    self.render_widget.add_mesh(
                        bvh_mesh, color=self.sphere_wire_color, style="wireframe",
                        line_width=self.sphere_wire_width, ambient=0.5, reset_camera=False
                    )

            if self.highlight_node_tri:
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

            if self.show_ray and hasattr(self.data.ray_data, "ray"):
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

        if self.first_render and self.data.full_scene_mesh.n_points > 0:
            self.render_widget.reset_camera()
            self.first_render = False

        self.render_widget.update()

# ====================== 主函数（3. 修改：更新文件路径） ======================
if __name__ == "__main__":
    # 关键修改：文件路径指向Cache/场景名目录下的Protobuf文件
    # 需根据你的项目实际路径调整（示例路径为"../Cache/inroom1/"，与C++输出对齐）
    AABB_BVH_PATH = "Cache/inroom1/aabb_bvh_structure.pb"
    SPHERE_BVH_PATH = "Cache/inroom1/sphere_bvh_structure.pb"
    OBJ_PATH = "obj/inroom1.obj"  # OBJ路径不变
    RAY_PATH = "Cache/inroom1/ray_hit_data.pb"

    # 检查文件存在性（不变）
    for fp in [AABB_BVH_PATH, SPHERE_BVH_PATH, OBJ_PATH, RAY_PATH]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"文件不存在：{fp} → 请检查路径！")

    # 启动应用（不变）
    data_loader = BVHDataLoader(AABB_BVH_PATH, SPHERE_BVH_PATH, OBJ_PATH, RAY_PATH)
    app = QApplication(sys.argv)
    window = BVHMainWindow(data_loader)
    window.show()
    sys.exit(app.exec_())