#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <set>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <limits>
#include <ctime>
#include <mutex>
#include <functional>
#include <filesystem>
#include <google/protobuf/repeated_field.h>
#include <atomic> // 新增：原子变量支持
#include "BVHData.pb.h"

// 全局常量（新增 kMaxThreadCount，优化并行控制）
const double kEps = 1e-8;               // 计算阈值
const double kRayMaxDistance = 1e6;     // 射线最大检测距离
const double kAABBEpsilon = 1e-6;       // AABB膨胀epsilon
const int kMaxBVHDepth = 64;            // BVH最大深度
const int kMaxLeafSize = 8;             // 叶子节点最大三角数
const int kSAHBucketCount = 16;         // SAH桶数量
const int kParallelThreshold = 200;     // 并行阈值
const int kMaxThreadCount = 8;          // 最大并行线程数（避免超核心开销）

// 数值合法性校验
inline bool IsValidNumber(double val) {
    return !std::isnan(val) && !std::isinf(val);
}
inline double ClampValidNumber(double val, double defaultVal = 0.0) {
    return IsValidNumber(val) ? val : defaultVal;
}

// 进度条类（线程安全）- 完整声明
class ProgressBar {
public:
    // 声明三个参数的构造函数（默认updateInterval=100）
    ProgressBar(int totalSteps, const std::string& title, int updateInterval = 100);
    void Update(int steps = 1);
    void Finish();

private:
    void Display();
    int total_steps_;
    int current_steps_;
    std::string title_;
    std::mutex mtx_;
    bool is_finished_;
    int update_interval_; // 新增：进度条更新间隔（步数）
};

// 回调函数类型
typedef std::function<void(int steps)> BVHProgressCallback;
typedef std::function<void(int steps)> RayProgressCallback;

// 全局工具函数声明
std::string GetRuntimeTimestamp();
std::string GetFileLastModifyTime(const std::string& filePath);
bool CreateDirectoryRecursive(const std::string& dirPath);
int CountFileLines(const std::string& filePath);

// 三维点结构
struct Point3D {
    double x = 0.0, y = 0.0, z = 0.0;
    Point3D() = default;
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Point3D operator-(const Point3D& other) const { return { x - other.x, y - other.y, z - other.z }; }
    Point3D operator+(const Point3D& other) const { return { x + other.x, y + other.y, z + other.z }; }
    Point3D operator*(double scalar) const { return { x * scalar, y * scalar, z * scalar }; }
    double Length() const { return sqrt(x * x + y * y + z * z); }
    Point3D Normalize() const {
        double len = Length();
        return len > kEps ? Point3D(x / len, y / len, z / len) : Point3D(0, 0, 0);
    }
};

// 三角形结构
struct Triangle3D {
    int p1 = 0, p2 = 0, p3 = 0;
    Point3D normal;
    Triangle3D() = default;
    Triangle3D(int p1_, int p2_, int p3_, const Point3D& n_) : p1(p1_), p2(p2_), p3(p3_), normal(n_) {}
};

// 场景结构
struct Scenario3D {
    std::vector<Point3D> points;
    std::vector<Triangle3D> triangles;
    Scenario3D() = default;
    ~Scenario3D() { points.clear(); triangles.clear(); }
};

// 单个碰撞结果
struct SingleHitResult {
    double distance = kRayMaxDistance;
    int triangleIndex = -1;
    Point3D hitPoint;
    bool hasHit = false;
};

// 射线碰撞结果
struct RayIntersectResult {
    bool hasAnyHit = false;
    std::vector<SingleHitResult> allHits;
    std::mutex mtx;

    RayIntersectResult(RayIntersectResult&& other) noexcept
        : hasAnyHit(other.hasAnyHit), allHits(std::move(other.allHits)) {}
    RayIntersectResult& operator=(RayIntersectResult&& other) noexcept {
        if (this != &other) {
            hasAnyHit = other.hasAnyHit;
            allHits = std::move(other.allHits);
        }
        return *this;
    }
    RayIntersectResult() = default;
    RayIntersectResult(const RayIntersectResult&) = delete;
    RayIntersectResult& operator=(const RayIntersectResult&) = delete;

    void AddHit(const SingleHitResult& hit) {
        if (hit.hasHit) {
            std::lock_guard<std::mutex> lock(mtx);
            allHits.push_back(hit);
            hasAnyHit = true;
            std::sort(allHits.begin(), allHits.end(), [](const SingleHitResult& a, const SingleHitResult& b) {
                return a.distance < b.distance;
                });
        }
    }

    void MergeLocalHits(const std::vector<SingleHitResult>& localHits) {
        if (localHits.empty()) return;
        std::lock_guard<std::mutex> lock(mtx);
        allHits.insert(allHits.end(), localHits.begin(), localHits.end());
        hasAnyHit = true;
        std::sort(allHits.begin(), allHits.end(), [](const SingleHitResult& a, const SingleHitResult& b) {
            return a.distance < b.distance;
            });
    }

    SingleHitResult GetClosestHit() const {
        return hasAnyHit && !allHits.empty() ? allHits[0] : SingleHitResult();
    }
};

// AABB包围盒
struct AABB {
    Point3D min;
    Point3D max;
    double surfaceArea = 0.0;
    std::vector<int> triangleIndices;

    AABB() = default;
    AABB(const Point3D& min_, const Point3D& max_) : min(min_), max(max_) {
        double dx = std::max(kAABBEpsilon, max.x - min.x);
        double dy = std::max(kAABBEpsilon, max.y - min.y);
        double dz = std::max(kAABBEpsilon, max.z - min.z);
        surfaceArea = 2.0 * (dx * dy + dy * dz + dz * dx);
    }

    static AABB Merge(const AABB& a, const AABB& b) {
        return AABB(
            Point3D(std::min(a.min.x, b.min.x), std::min(a.min.y, b.min.y), std::min(a.min.z, b.min.z)),
            Point3D(std::max(a.max.x, b.max.x), std::max(a.max.y, b.max.y), std::max(a.max.z, b.max.z))
        );
    }
};

// 包围球
struct BoundingSphere {
    Point3D center;
    double radius = 0.0;
    std::vector<int> triangleIndices;

    BoundingSphere() = default;
    BoundingSphere(const Point3D& c_, double r_) : center(c_), radius(r_) {}

    static BoundingSphere Merge(const BoundingSphere& a, const BoundingSphere& b) {
        Point3D dir = b.center - a.center;
        double dist = dir.Length();
        if (dist <= a.radius + b.radius) {
            if (a.radius >= dist + b.radius) return a;
            if (b.radius >= dist + a.radius) return b;
        }
        double newRadius = (a.radius + dist + b.radius) / 2.0;
        Point3D newCenter = a.center + dir.Normalize() * (newRadius - a.radius);
        return BoundingSphere(newCenter, newRadius);
    }
};

// BVH节点（模板类）
template <typename BoundType>
struct BVHNode {
    bool isLeaf = false;
    BoundType bound;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    int depth = 0;
    mutable int nodeId = -1;

    BVHNode() = default;
    BVHNode(const BVHNode&) = delete;
    BVHNode& operator=(const BVHNode&) = delete;
    BVHNode(BVHNode&&) = default;
    BVHNode& operator=(BVHNode&&) = default;
};

// 计时工具
using TimePoint = std::chrono::high_resolution_clock::time_point;
inline TimePoint GetCurrentTime() { return std::chrono::high_resolution_clock::now(); }
inline double CalculateElapsedMs(const TimePoint& start, const TimePoint& end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
}

// 几何工具函数（新增 DistanceSq 用于平方距离计算）
namespace GeometryUtils {
    double Dot(const Point3D& a, const Point3D& b);
    Point3D Cross(const Point3D& a, const Point3D& b);
    double Distance(const Point3D& a, const Point3D& b);
    double DistanceSq(const Point3D& a, const Point3D& b); // 新增：平方距离（避免sqrt）
    Point3D Normalize(const Point3D& a);
    Point3D Sub(const Point3D& a, const Point3D& b);
    Point3D Add(const Point3D& a, const Point3D& b);
    Point3D Mul(const Point3D& a, double d);
    std::vector<Point3D> GenerateRayPoints(const Point3D& origin, const Point3D& dir, double maxDist, int sampleCount = 50);
}

// 模型导入
namespace ModelImporter {
    bool LoadOBJ(const std::string& filePath, Scenario3D& scene, ProgressBar* progressBar = nullptr);
    void TriangulatePolygon(const std::vector<int>& indices, std::vector<std::tuple<int, int, int>>& triangles);
}

// BVH加速结构（更新函数声明，适配优化逻辑）
namespace BVHAccelerator {
    // AABB-BVH构建
    AABB BuildAABBForTriangle(const Scenario3D& scene, int triIndex);
    AABB BuildAABBForTriangles(const Scenario3D& scene, const std::vector<int>& triIndices);
    std::unique_ptr<BVHNode<AABB>> BuildAABBBVHRecursive(const Scenario3D& scene, const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback, std::atomic<int>& processedTriangles); // 新增 atomic 参数
    std::unique_ptr<BVHNode<AABB>> BuildAABBBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback = nullptr);

    // Sphere-BVH构建（替换 RitterMinSphere 为优化版本）
    BoundingSphere RitterMinSphereOptimized(const Scenario3D& scene, const std::vector<int>& triIndices);
    BoundingSphere BuildSphereForTriangles(const Scenario3D& scene, const std::vector<int>& triIndices);
    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVHRecursive(const Scenario3D& scene, const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback, std::atomic<int>& processedTriangles); // 新增 atomic 参数
    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback = nullptr);

    // 新增：计算节点到射线的最小距离（用于距离优先遍历）
    double CalculateNodeMinDistance(const Point3D& rayOrigin, const Point3D& rayDir, const AABB& aabb);
    double CalculateNodeMinDistance(const Point3D& rayOrigin, const Point3D& rayDir, const BoundingSphere& sphere);

    // 射线检测（更新声明，适配优化逻辑）
    bool RayIntersectAABB_Box(const Point3D& rayOrigin, const Point3D& rayDir, const Point3D& rayDirInv, const AABB& aabb); // 新增 rayDirInv 参数
    bool RayIntersectSphere(const Point3D& rayOrigin, const Point3D& rayDir, const BoundingSphere& sphere);
    SingleHitResult RayIntersectTriangle(const Scenario3D& scene, int triIndex, const Point3D& rayOrigin, const Point3D& rayDir, double& tMin); // 新增 tMin 参数
    void TraverseAABBBVH(const Scenario3D& scene, const BVHNode<AABB>* node, const Point3D& rayOrigin, const Point3D& rayDir, const Point3D& rayDirInv, RayIntersectResult& result, std::atomic<double>& tMin, const RayProgressCallback& progressCallback, std::mutex& progressMtx); // 新增 rayDirInv 和 atomic tMin
    void TraverseSphereBVH(const Scenario3D& scene, const BVHNode<BoundingSphere>* node, const Point3D& rayOrigin, const Point3D& rayDir, RayIntersectResult& result, std::atomic<double>& tMin, const RayProgressCallback& progressCallback, std::mutex& progressMtx); // 新增 atomic tMin
    RayIntersectResult RayIntersectAABB(const Scenario3D& scene, const BVHNode<AABB>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback = nullptr);
    RayIntersectResult RayIntersectSphere(const Scenario3D& scene, const BVHNode<BoundingSphere>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback = nullptr);

    // Protobuf导出
    void FillAABBNodeToProto(const BVHNode<AABB>* node, int& nodeId, BVHData::BVHStructureProto* bvhProto);
    void FillSphereNodeToProto(const BVHNode<BoundingSphere>* node, int& nodeId, BVHData::BVHStructureProto* bvhProto);
    void ExportBVHToProtobuf(const std::string& filePath, const BVHNode<AABB>* root, const std::string& bvhType);
    void ExportBVHToProtobuf(const std::string& filePath, const BVHNode<BoundingSphere>* root, const std::string& bvhType);

    // BVH节点数统计
    template <typename BoundType>
    int CountBVHNodes(const std::unique_ptr<BVHNode<BoundType>>& root);

    // BVH缓存保存/加载
    template <typename BoundType>
    bool SaveBVHCached(const std::string& cacheFilePath, const std::unique_ptr<BVHNode<BoundType>>& bvhRoot, const std::string& objFilePath, const std::string& objFileMTime, int vertexCount, int triangleCount, const std::string& bvhType, double buildTimeMs);
    template <typename BoundType>
    std::unique_ptr<BVHNode<BoundType>> ReconstructBVHNode(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>& nodes, int& nodeId);
    template <typename BoundType>
    std::unique_ptr<BVHNode<BoundType>> LoadBVHCached(const std::string& cacheFilePath, const std::string& objFilePath, const std::string& objFileMTime, int vertexCount, int triangleCount, bool& isCacheValid, double& outBuildTime);
}

// 可视化工具
namespace Visualizer {
    void ShowRayAndHitDetail(const Point3D& rayOrigin, const Point3D& rayDir, const RayIntersectResult& result);
    void ExportRayAndHitToProtobuf(const std::string& filePath, const Point3D& rayOrigin, const Point3D& rayDir, const RayIntersectResult& result);
    void ShowTimeComparison(double aabbInit, double sphereInit, double aabbDetect, double sphereDetect);
    void ShowHitResult(const std::string& method, const RayIntersectResult& result);
}