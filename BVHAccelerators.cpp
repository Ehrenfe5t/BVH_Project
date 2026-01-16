#define _CRT_SECURE_NO_WARNINGS
#include "Core.h"
#include <omp.h>
#include <google/protobuf/repeated_field.h>

namespace BVHAccelerator {
    constexpr double kAABBEpsilon = 1e-6;
    constexpr double kEps = 1e-8;
    constexpr double kRayMaxDistance = 1e6;
    constexpr int kMaxLeafSize = 8;
    constexpr int kMaxBVHDepth = 64;
    constexpr int kSAHBucketCount = 16;
    constexpr int kParallelThreshold = 200;

    // SAH核心工具函数
    double GetTriangleCentroidAxis(const Scenario3D& scene, int triIdx, int axis) {
        const auto& tri = scene.triangles[triIdx];
        const auto& p1 = scene.points[tri.p1];
        const auto& p2 = scene.points[tri.p2];
        const auto& p3 = scene.points[tri.p3];
        if (axis == 0) return (p1.x + p2.x + p3.x) / 3.0;
        if (axis == 1) return (p1.y + p2.y + p3.y) / 3.0;
        return (p1.z + p2.z + p3.z) / 3.0;
    }

    double CalculateAABBSurfaceArea(const AABB& aabb) {
        double dx = aabb.max.x - aabb.min.x;
        double dy = aabb.max.y - aabb.min.y;
        double dz = aabb.max.z - aabb.min.z;
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }

    double CalculateSAHCost(const AABB& parentAABB,
        const AABB& leftAABB, size_t leftCount,
        const AABB& rightAABB, size_t rightCount) {
        double parentArea = CalculateAABBSurfaceArea(parentAABB);
        if (parentArea < kEps) return std::numeric_limits<double>::max();
        double leftArea = CalculateAABBSurfaceArea(leftAABB);
        double rightArea = CalculateAABBSurfaceArea(rightAABB);
        return (leftArea * leftCount + rightArea * rightCount) / parentArea;
    }

    void GetCentroidRange(const Scenario3D& scene, const std::vector<int>& triIndices, int axis,
        double& minCentroid, double& maxCentroid) {
        minCentroid = std::numeric_limits<double>::max();
        maxCentroid = -std::numeric_limits<double>::max();
        for (int triIdx : triIndices) {
            double c = GetTriangleCentroidAxis(scene, triIdx, axis);
            minCentroid = std::min(minCentroid, c);
            maxCentroid = std::max(maxCentroid, c);
        }
        if (maxCentroid - minCentroid < kEps) {
            minCentroid -= kEps;
            maxCentroid += kEps;
        }
    }

    bool FindSAHOptimalSplit(const Scenario3D& scene, const std::vector<int>& triIndices, int axis,
        const AABB& parentAABB,
        std::vector<int>& outLeftIndices, std::vector<int>& outRightIndices,
        double& outBestCost) {
        outLeftIndices.clear();
        outRightIndices.clear();
        outBestCost = std::numeric_limits<double>::max();
        double minCentroid, maxCentroid;
        GetCentroidRange(scene, triIndices, axis, minCentroid, maxCentroid);
        double centroidExtent = maxCentroid - minCentroid;
        if (centroidExtent < kEps) return false;
        std::vector<int> sortedTriIndices = triIndices;
        std::sort(sortedTriIndices.begin(), sortedTriIndices.end(),
            [&](int a, int b) {
                return GetTriangleCentroidAxis(scene, a, axis) < GetTriangleCentroidAxis(scene, b, axis);
            });
        for (int splitIdx = 1; splitIdx < sortedTriIndices.size(); ++splitIdx) {
            std::vector<int> leftIndices(sortedTriIndices.begin(), sortedTriIndices.begin() + splitIdx);
            std::vector<int> rightIndices(sortedTriIndices.begin() + splitIdx, sortedTriIndices.end());
            if (leftIndices.empty() || rightIndices.empty()) continue;
            AABB leftAABB = BuildAABBForTriangles(scene, leftIndices);
            AABB rightAABB = BuildAABBForTriangles(scene, rightIndices);
            double cost = CalculateSAHCost(parentAABB, leftAABB, leftIndices.size(), rightAABB, rightIndices.size());
            if (cost < outBestCost) {
                outBestCost = cost;
                outLeftIndices = leftIndices;
                outRightIndices = rightIndices;
            }
        }
        return !outLeftIndices.empty() && !outRightIndices.empty();
    }

    bool SelectSAHOptimalAxis(const Scenario3D& scene, const std::vector<int>& triIndices,
        const AABB& parentAABB,
        std::vector<int>& outLeftIndices, std::vector<int>& outRightIndices) {
        double bestCost = std::numeric_limits<double>::max();
        int bestAxis = -1;
        std::vector<int> bestLeft, bestRight;
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<int> leftIndices, rightIndices;
            double axisCost;
            if (FindSAHOptimalSplit(scene, triIndices, axis, parentAABB, leftIndices, rightIndices, axisCost)) {
                if (axisCost < bestCost) {
                    bestCost = axisCost;
                    bestAxis = axis;
                    bestLeft = leftIndices;
                    bestRight = rightIndices;
                }
            }
        }
        if (bestAxis == -1) return false;
        outLeftIndices = bestLeft;
        outRightIndices = bestRight;
        return true;
    }

    // 工具函数
    AABB BuildAABBForTriangle(const Scenario3D& scene, int triIndex) {
        const auto& tri = scene.triangles[triIndex];
        const auto& p1 = scene.points[tri.p1];
        const auto& p2 = scene.points[tri.p2];
        const auto& p3 = scene.points[tri.p3];
        Point3D minP(
            std::min({ p1.x, p2.x, p3.x }) - kAABBEpsilon,
            std::min({ p1.y, p2.y, p3.y }) - kAABBEpsilon,
            std::min({ p1.z, p2.z, p3.z }) - kAABBEpsilon
        );
        Point3D maxP(
            std::max({ p1.x, p2.x, p3.x }) + kAABBEpsilon,
            std::max({ p1.y, p2.y, p3.y }) + kAABBEpsilon,
            std::max({ p1.z, p2.z, p3.z }) + kAABBEpsilon
        );
        AABB aabb(minP, maxP);
        aabb.triangleIndices.push_back(triIndex);
        return aabb;
    }

    AABB BuildAABBForTriangles(const Scenario3D& scene, const std::vector<int>& triIndices) {
        if (triIndices.empty()) return AABB();
        AABB aabb = BuildAABBForTriangle(scene, triIndices[0]);
        for (size_t i = 1; i < triIndices.size(); ++i) {
            AABB triAABB = BuildAABBForTriangle(scene, triIndices[i]);
            aabb.min.x = std::min(aabb.min.x, triAABB.min.x);
            aabb.min.y = std::min(aabb.min.y, triAABB.min.y);
            aabb.min.z = std::min(aabb.min.z, triAABB.min.z);
            aabb.max.x = std::max(aabb.max.x, triAABB.max.x);
            aabb.max.y = std::max(aabb.max.y, triAABB.max.y);
            aabb.max.z = std::max(aabb.max.z, triAABB.max.z);
        }
        aabb.triangleIndices = triIndices;
        return aabb;
    }

    BoundingSphere RitterMinSphere(const Scenario3D& scene, const std::vector<int>& triIndices) {
        if (triIndices.empty()) return BoundingSphere();
        Point3D minP(1e9, 1e9, 1e9), maxP(-1e9, -1e9, -1e9);
        for (int triIdx : triIndices) {
            const auto& tri = scene.triangles[triIdx];
            const auto& p1 = scene.points[tri.p1];
            const auto& p2 = scene.points[tri.p2];
            const auto& p3 = scene.points[tri.p3];
            minP = Point3D(std::min({ minP.x, p1.x, p2.x, p3.x }),
                std::min({ minP.y, p1.y, p2.y, p3.y }),
                std::min({ minP.z, p1.z, p2.z, p3.z }));
            maxP = Point3D(std::max({ maxP.x, p1.x, p2.x, p3.x }),
                std::max({ maxP.y, p1.y, p2.y, p3.y }),
                std::max({ maxP.z, p1.z, p2.z, p3.z }));
        }
        Point3D p1 = minP;
        Point3D p2 = p1;
        double maxDist = 0;
        for (int triIdx : triIndices) {
            const auto& tri = scene.triangles[triIdx];
            const auto& pts = { scene.points[tri.p1], scene.points[tri.p2], scene.points[tri.p3] };
            for (const auto& p : pts) {
                double dist = GeometryUtils::Distance(p1, p);
                if (dist > maxDist) { maxDist = dist; p2 = p; }
            }
        }
        Point3D center = GeometryUtils::Add(p1, p2);
        center = GeometryUtils::Mul(center, 0.5);
        double radius = GeometryUtils::Distance(p1, center);
        for (int triIdx : triIndices) {
            const auto& tri = scene.triangles[triIdx];
            const auto& pts = { scene.points[tri.p1], scene.points[tri.p2], scene.points[tri.p3] };
            for (const auto& p : pts) {
                double dist = GeometryUtils::Distance(center, p);
                if (dist > radius) {
                    double newRadius = (radius + dist) * 0.5;
                    Point3D dir = GeometryUtils::Sub(p, center);
                    dir = GeometryUtils::Normalize(dir);
                    center = GeometryUtils::Add(center, GeometryUtils::Mul(dir, newRadius - radius));
                    radius = newRadius;
                }
            }
        }
        return BoundingSphere(center, radius + kAABBEpsilon);
    }

    BoundingSphere BuildSphereForTriangles(const Scenario3D& scene, const std::vector<int>& triIndices) {
        if (triIndices.empty()) return BoundingSphere();
        BoundingSphere sphere = RitterMinSphere(scene, triIndices);
        sphere.triangleIndices = triIndices;
        return sphere;
    }

    // BVH构建（带进度回调）
    std::unique_ptr<BVHNode<AABB>> BuildAABBBVHRecursive(const Scenario3D& scene,
        const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback) {
        auto node = std::make_unique<BVHNode<AABB>>();
        node->bound = BuildAABBForTriangles(scene, triIndices);
        node->depth = depth;

        if (triIndices.size() <= kMaxLeafSize || depth >= kMaxBVHDepth) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            return node;
        }

        std::vector<int> leftIndices, rightIndices;
        bool splitSuccess = SelectSAHOptimalAxis(scene, triIndices, node->bound, leftIndices, rightIndices);
        if (!splitSuccess || leftIndices.empty() || rightIndices.empty()) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            return node;
        }

        node->left = BuildAABBBVHRecursive(scene, leftIndices, depth + 1, progressCallback);
        node->right = BuildAABBBVHRecursive(scene, rightIndices, depth + 1, progressCallback);
        return node;
    }

    std::unique_ptr<BVHNode<AABB>> BuildAABBBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback) {
        std::vector<int> triIndices(scene.triangles.size());
        std::iota(triIndices.begin(), triIndices.end(), 0);
        return BuildAABBBVHRecursive(scene, triIndices, 0, progressCallback);
    }

    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVHRecursive(const Scenario3D& scene,
        const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback) {
        auto node = std::make_unique<BVHNode<BoundingSphere>>();
        node->bound = BuildSphereForTriangles(scene, triIndices);
        node->depth = depth;

        if (triIndices.size() <= kMaxLeafSize || depth >= kMaxBVHDepth) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            return node;
        }

        AABB sphereAABB;
        sphereAABB.min = Point3D(node->bound.center.x - node->bound.radius,
            node->bound.center.y - node->bound.radius,
            node->bound.center.z - node->bound.radius);
        sphereAABB.max = Point3D(node->bound.center.x + node->bound.radius,
            node->bound.center.y + node->bound.radius,
            node->bound.center.z + node->bound.radius);

        std::vector<int> leftIndices, rightIndices;
        bool splitSuccess = SelectSAHOptimalAxis(scene, triIndices, sphereAABB, leftIndices, rightIndices);
        if (!splitSuccess || leftIndices.empty() || rightIndices.empty()) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            return node;
        }

        node->left = BuildSphereBVHRecursive(scene, leftIndices, depth + 1, progressCallback);
        node->right = BuildSphereBVHRecursive(scene, rightIndices, depth + 1, progressCallback);
        return node;
    }

    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback) {
        std::vector<int> triIndices(scene.triangles.size());
        std::iota(triIndices.begin(), triIndices.end(), 0);
        return BuildSphereBVHRecursive(scene, triIndices, 0, progressCallback);
    }

    // 射线检测（带进度回调）
    bool RayIntersectAABB_Box(const Point3D& rayOrigin, const Point3D& rayDir, const AABB& aabb) {
        auto safeDiv = [](double num, double denom) -> double {
            return denom == 0 ? (num > 0 ? kRayMaxDistance : -kRayMaxDistance) : num / denom;
            };
        double tMinX = safeDiv(aabb.min.x - rayOrigin.x, rayDir.x);
        double tMaxX = safeDiv(aabb.max.x - rayOrigin.x, rayDir.x);
        if (tMinX > tMaxX) std::swap(tMinX, tMaxX);
        double tMinY = safeDiv(aabb.min.y - rayOrigin.y, rayDir.y);
        double tMaxY = safeDiv(aabb.max.y - rayOrigin.y, rayDir.y);
        if (tMinY > tMaxY) std::swap(tMinY, tMaxY);
        if (tMinX > tMaxY || tMinY > tMaxX) return false;
        tMinX = std::max(tMinX, tMinY);
        tMaxX = std::min(tMaxX, tMaxY);
        double tMinZ = safeDiv(aabb.min.z - rayOrigin.z, rayDir.z);
        double tMaxZ = safeDiv(aabb.max.z - rayOrigin.z, rayDir.z);
        if (tMinZ > tMaxZ) std::swap(tMinZ, tMaxZ);
        if (tMinX > tMaxZ || tMinZ > tMaxX) return false;
        tMinX = std::max(tMinX, tMinZ);
        tMaxX = std::min(tMaxX, tMaxZ);
        return tMaxX >= tMinX && tMinX < kRayMaxDistance && tMaxX > kEps;
    }

    bool RayIntersectSphere(const Point3D& rayOrigin, const Point3D& rayDir, const BoundingSphere& sphere) {
        Point3D oc = GeometryUtils::Sub(sphere.center, rayOrigin);
        double a = GeometryUtils::Dot(rayDir, rayDir);
        if (a < kEps) {
            double distSq = GeometryUtils::Dot(oc, oc);
            return distSq <= sphere.radius * sphere.radius + kEps;
        }
        double b = GeometryUtils::Dot(oc, rayDir);
        double c = GeometryUtils::Dot(oc, oc) - sphere.radius * sphere.radius;
        double discriminant = b * b - a * c;
        if (discriminant < -kEps) return false;
        discriminant = std::max(discriminant, 0.0);
        double sqrtDisc = sqrt(discriminant);
        double t1 = (b - sqrtDisc) / a;
        double t2 = (b + sqrtDisc) / a;
        bool t1Valid = t1 > kEps && t1 < kRayMaxDistance;
        bool t2Valid = t2 > kEps && t2 < kRayMaxDistance;
        return t1Valid || t2Valid;
    }

    SingleHitResult RayIntersectTriangle(const Scenario3D& scene, int triIndex,
        const Point3D& rayOrigin, const Point3D& rayDir) {
        SingleHitResult hit;
        const auto& tri = scene.triangles[triIndex];
        const auto& p1 = scene.points[tri.p1];
        const auto& p2 = scene.points[tri.p2];
        const auto& p3 = scene.points[tri.p3];
        Point3D e1 = GeometryUtils::Sub(p2, p1);
        Point3D e2 = GeometryUtils::Sub(p3, p1);
        Point3D h = GeometryUtils::Cross(rayDir, e2);
        double a = GeometryUtils::Dot(e1, h);
        if (fabs(a) < kEps) return hit;
        double f = 1.0 / a;
        Point3D s = GeometryUtils::Sub(rayOrigin, p1);
        double u = f * GeometryUtils::Dot(s, h);
        if (u < 0.0 || u > 1.0) return hit;
        Point3D q = GeometryUtils::Cross(s, e1);
        double v = f * GeometryUtils::Dot(rayDir, q);
        if (v < 0.0 || u + v > 1.0) return hit;
        double t = f * GeometryUtils::Dot(e2, q);
        if (t > kEps && t < kRayMaxDistance) {
            hit.hasHit = true;
            hit.distance = t;
            hit.triangleIndex = triIndex;
            hit.hitPoint = GeometryUtils::Add(rayOrigin, GeometryUtils::Mul(rayDir, t));
        }
        return hit;
    }

    void TraverseAABBBVH(const Scenario3D& scene, const BVHNode<AABB>* node,
        const Point3D& rayOrigin, const Point3D& rayDir, RayIntersectResult& result, const RayProgressCallback& progressCallback, std::mutex& progressMtx) {
        if (!node || !RayIntersectAABB_Box(rayOrigin, rayDir, node->bound)) return;

        if (node->isLeaf) {
            const auto& triIndices = node->bound.triangleIndices;
            const int triCount = static_cast<int>(triIndices.size());

            if (triCount > kParallelThreshold) {
                thread_local std::vector<SingleHitResult> localHits;
                localHits.clear();
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
                for (int i = 0; i < triCount; ++i) {
                    int triIdx = triIndices[i];
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir);
                    if (hit.hasHit) localHits.push_back(hit);
                    if (progressCallback) {
                        std::lock_guard<std::mutex> lock(progressMtx);
                        progressCallback(1);
                    }
                }
                result.MergeLocalHits(localHits);
            }
            else {
                for (int triIdx : triIndices) {
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir);
                    if (hit.hasHit) result.AddHit(hit);
                    if (progressCallback) progressCallback(1);
                }
            }
            return;
        }

        TraverseAABBBVH(scene, node->left.get(), rayOrigin, rayDir, result, progressCallback, progressMtx);
        TraverseAABBBVH(scene, node->right.get(), rayOrigin, rayDir, result, progressCallback, progressMtx);
    }

    void TraverseSphereBVH(const Scenario3D& scene, const BVHNode<BoundingSphere>* node,
        const Point3D& rayOrigin, const Point3D& rayDir, RayIntersectResult& result, const RayProgressCallback& progressCallback, std::mutex& progressMtx) {
        if (!node || !RayIntersectSphere(rayOrigin, rayDir, node->bound)) return;

        if (node->isLeaf) {
            const auto& triIndices = node->bound.triangleIndices;
            const int triCount = static_cast<int>(triIndices.size());

            if (triCount > kParallelThreshold) {
                thread_local std::vector<SingleHitResult> localHits;
                localHits.clear();
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
                for (int i = 0; i < triCount; ++i) {
                    int triIdx = triIndices[i];
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir);
                    if (hit.hasHit) localHits.push_back(hit);
                    if (progressCallback) {
                        std::lock_guard<std::mutex> lock(progressMtx);
                        progressCallback(1);
                    }
                }
                result.MergeLocalHits(localHits);
            }
            else {
                for (int triIdx : triIndices) {
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir);
                    if (hit.hasHit) result.AddHit(hit);
                    if (progressCallback) progressCallback(1);
                }
            }
            return;
        }

        TraverseSphereBVH(scene, node->left.get(), rayOrigin, rayDir, result, progressCallback, progressMtx);
        TraverseSphereBVH(scene, node->right.get(), rayOrigin, rayDir, result, progressCallback, progressMtx);
    }

    RayIntersectResult RayIntersectAABB(const Scenario3D& scene, const BVHNode<AABB>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback) {
        RayIntersectResult result;
        std::mutex progressMtx;
        TraverseAABBBVH(scene, root, rayOrigin, rayDir, result, progressCallback, progressMtx);
        return result;
    }

    RayIntersectResult RayIntersectSphere(const Scenario3D& scene, const BVHNode<BoundingSphere>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback) {
        RayIntersectResult result;
        std::mutex progressMtx;
        TraverseSphereBVH(scene, root, rayOrigin, rayDir, result, progressCallback, progressMtx);
        return result;
    }

    // Protobuf导出
    void FillAABBNodeToProto(const BVHNode<AABB>* node, int& nodeId, BVHData::BVHStructureProto* bvhProto) {
        if (!node) return;
        node->nodeId = nodeId++;
        auto* nodeProto = bvhProto->add_nodes();
        nodeProto->set_node_id(node->nodeId);
        nodeProto->set_depth(node->depth);
        nodeProto->set_node_type(node->isLeaf ? "leaf" : "internal");
        nodeProto->set_bound_type("AABB");
        auto* aabbProto = nodeProto->mutable_aabb_bound();
        auto* minProto = aabbProto->mutable_min();
        minProto->set_x(node->bound.min.x);
        minProto->set_y(node->bound.min.y);
        minProto->set_z(node->bound.min.z);
        auto* maxProto = aabbProto->mutable_max();
        maxProto->set_x(node->bound.max.x);
        maxProto->set_y(node->bound.max.y);
        maxProto->set_z(node->bound.max.z);
        aabbProto->set_surface_area(node->bound.surfaceArea);
        for (int triIdx : node->bound.triangleIndices) {
            nodeProto->add_triangle_indices(triIdx);
        }
        FillAABBNodeToProto(node->left.get(), nodeId, bvhProto);
        FillAABBNodeToProto(node->right.get(), nodeId, bvhProto);
    }

    void FillSphereNodeToProto(const BVHNode<BoundingSphere>* node, int& nodeId, BVHData::BVHStructureProto* bvhProto) {
        if (!node) return;
        node->nodeId = nodeId++;
        auto* nodeProto = bvhProto->add_nodes();
        nodeProto->set_node_id(node->nodeId);
        nodeProto->set_depth(node->depth);
        nodeProto->set_node_type(node->isLeaf ? "leaf" : "internal");
        nodeProto->set_bound_type("Sphere");
        auto* sphereProto = nodeProto->mutable_sphere_bound();
        auto* centerProto = sphereProto->mutable_center();
        centerProto->set_x(node->bound.center.x);
        centerProto->set_y(node->bound.center.y);
        centerProto->set_z(node->bound.center.z);
        sphereProto->set_radius(node->bound.radius);
        for (int triIdx : node->bound.triangleIndices) {
            nodeProto->add_triangle_indices(triIdx);
        }
        FillSphereNodeToProto(node->left.get(), nodeId, bvhProto);
        FillSphereNodeToProto(node->right.get(), nodeId, bvhProto);
    }

    void ExportBVHToProtobuf(const std::string& filePath, const BVHNode<AABB>* root, const std::string& bvhType) {
        if (!root) {
            std::cerr << "错误：BVH根节点为空，无法导出" << std::endl;
            return;
        }
        try {
            BVHData::BVHStructureProto bvhProto;
            auto* metadata = bvhProto.mutable_metadata();
            metadata->set_bvh_type(bvhType);
            metadata->set_export_time(GetRuntimeTimestamp());
            metadata->set_node_count(0);
            int nodeId = 0;
            FillAABBNodeToProto(root, nodeId, &bvhProto);
            metadata->set_node_count(nodeId);
            std::ofstream file(filePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "错误：无法导出BVH节点信息至 " << filePath << std::endl;
                return;
            }
            if (!bvhProto.SerializeToOstream(&file)) {
                std::cerr << "错误：AABB-BVH Protobuf序列化失败" << std::endl;
            }
            file.close();
            std::cout << "  " << bvhType << " structure exported to: " << filePath << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "错误：AABB-BVH Protobuf导出失败 - " << e.what() << std::endl;
        }
    }

    void ExportBVHToProtobuf(const std::string& filePath, const BVHNode<BoundingSphere>* root, const std::string& bvhType) {
        if (!root) {
            std::cerr << "错误：BVH根节点为空，无法导出" << std::endl;
            return;
        }
        try {
            BVHData::BVHStructureProto bvhProto;
            auto* metadata = bvhProto.mutable_metadata();
            metadata->set_bvh_type(bvhType);
            metadata->set_export_time(GetRuntimeTimestamp());
            metadata->set_node_count(0);
            int nodeId = 0;
            FillSphereNodeToProto(root, nodeId, &bvhProto);
            metadata->set_node_count(nodeId);
            std::ofstream file(filePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "错误：无法导出BVH节点信息至 " << filePath << std::endl;
                return;
            }
            if (!bvhProto.SerializeToOstream(&file)) {
                std::cerr << "错误：包围球-BVH Protobuf序列化失败" << std::endl;
            }
            file.close();
            std::cout << "  " << bvhType << " structure exported to: " << filePath << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "错误：包围球-BVH Protobuf导出失败 - " << e.what() << std::endl;
        }
    }

    // BVH节点数统计
    template <typename BoundType>
    int CountBVHNodes(const std::unique_ptr<BVHNode<BoundType>>& root) {
        if (!root) return 0;
        return 1 + CountBVHNodes(root->left) + CountBVHNodes(root->right);
    }

    // BVH缓存保存/加载
    template <typename BoundType>
    bool SaveBVHCached(const std::string& cacheFilePath,
        const std::unique_ptr<BVHNode<BoundType>>& bvhRoot,
        const std::string& objFilePath,
        const std::string& objFileMTime,
        int vertexCount,
        int triangleCount,
        const std::string& bvhType,
        double buildTimeMs) {
        try {
            BVHData::BVHCachedDataProto cachedData;
            auto* sceneInfo = cachedData.mutable_scene_info();
            sceneInfo->set_obj_file_path(objFilePath);
            sceneInfo->set_obj_file_mtime(objFileMTime);
            sceneInfo->set_triangle_count(triangleCount);
            sceneInfo->set_vertex_count(vertexCount);

            auto* cacheMeta = cachedData.mutable_cache_meta();
            cacheMeta->set_cache_time(GetRuntimeTimestamp());
            cacheMeta->set_build_time_ms(buildTimeMs);
            cacheMeta->set_bvh_type(bvhType);
            cacheMeta->set_optimize_method("SAH-Optimized");

            auto* bvhStruct = cachedData.mutable_bvh_data();
            auto* bvhMeta = bvhStruct->mutable_metadata();
            bvhMeta->set_bvh_type(bvhType);
            bvhMeta->set_export_time(GetRuntimeTimestamp());
            bvhMeta->set_node_count(0);

            int nodeId = 0;
            if constexpr (std::is_same_v<BoundType, AABB>) {
                FillAABBNodeToProto(bvhRoot.get(), nodeId, bvhStruct);
            }
            else if constexpr (std::is_same_v<BoundType, BoundingSphere>) {
                FillSphereNodeToProto(bvhRoot.get(), nodeId, bvhStruct);
            }
            bvhMeta->set_node_count(nodeId);

            std::filesystem::path cacheDir = std::filesystem::path(cacheFilePath).parent_path();
            std::filesystem::create_directories(cacheDir);
            std::ofstream file(cacheFilePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "错误：无法创建BVH缓存文件 " << cacheFilePath << std::endl;
                return false;
            }
            if (!cachedData.SerializeToOstream(&file)) {
                std::cerr << "错误：BVH缓存序列化失败 " << cacheFilePath << std::endl;
                return false;
            }
            file.close();
            std::cout << "  BVH缓存已保存至：" << cacheFilePath << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "错误：保存BVH缓存失败 - " << e.what() << std::endl;
            return false;
        }
    }

    template <typename BoundType>
    std::unique_ptr<BVHNode<BoundType>> ReconstructBVHNode(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>& nodes, int& nodeId) {
        if (nodeId >= nodes.size()) return nullptr;
        const auto& nodeProto = nodes.Get(nodeId);
        auto node = std::make_unique<BVHNode<BoundType>>();
        node->nodeId = nodeId;
        node->depth = nodeProto.depth();
        node->isLeaf = (nodeProto.node_type() == "leaf");

        if constexpr (std::is_same_v<BoundType, AABB>) {
            const auto& aabbProto = nodeProto.aabb_bound();
            Point3D minP(aabbProto.min().x(), aabbProto.min().y(), aabbProto.min().z());
            Point3D maxP(aabbProto.max().x(), aabbProto.max().y(), aabbProto.max().z());
            node->bound = AABB(minP, maxP);
            node->bound.surfaceArea = aabbProto.surface_area();
            for (int i = 0; i < nodeProto.triangle_indices_size(); ++i) {
                node->bound.triangleIndices.push_back(nodeProto.triangle_indices(i));
            }
        }
        else if constexpr (std::is_same_v<BoundType, BoundingSphere>) {
            const auto& sphereProto = nodeProto.sphere_bound();
            Point3D center(sphereProto.center().x(), sphereProto.center().y(), sphereProto.center().z());
            node->bound = BoundingSphere(center, sphereProto.radius());
            for (int i = 0; i < nodeProto.triangle_indices_size(); ++i) {
                node->bound.triangleIndices.push_back(nodeProto.triangle_indices(i));
            }
        }

        if (!node->isLeaf) {
            nodeId++;
            node->left = ReconstructBVHNode<BoundType>(nodes, nodeId);
            nodeId++;
            node->right = ReconstructBVHNode<BoundType>(nodes, nodeId);
        }

        return node;
    }

    template <typename BoundType>
    std::unique_ptr<BVHNode<BoundType>> LoadBVHCached(const std::string& cacheFilePath,
        const std::string& objFilePath,
        const std::string& objFileMTime,
        int vertexCount,
        int triangleCount,
        bool& isCacheValid,
        double& outBuildTime) {
        isCacheValid = false;
        outBuildTime = 0.0;
        try {
            std::ifstream file(cacheFilePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "提示：BVH缓存文件不存在 " << cacheFilePath << std::endl;
                return nullptr;
            }

            BVHData::BVHCachedDataProto cachedData;
            if (!cachedData.ParseFromIstream(&file)) {
                std::cerr << "错误：BVH缓存文件解析失败 " << cacheFilePath << std::endl;
                file.close();
                return nullptr;
            }
            file.close();

            const auto& sceneInfo = cachedData.scene_info();
            if (sceneInfo.obj_file_path() != objFilePath ||
                sceneInfo.obj_file_mtime() != objFileMTime ||
                sceneInfo.vertex_count() != vertexCount ||
                sceneInfo.triangle_count() != triangleCount) {
                std::cerr << "提示：BVH缓存无效（场景已修改），将重新构建" << std::endl;
                return nullptr;
            }

            // 提取缓存中的初次构建时间
            outBuildTime = cachedData.cache_meta().build_time_ms();

            const auto& bvhStruct = cachedData.bvh_data();
            int nodeId = 0;
            auto bvhRoot = ReconstructBVHNode<BoundType>(bvhStruct.nodes(), nodeId);
            if (!bvhRoot) {
                std::cerr << "错误：从缓存重建BVH失败 " << cacheFilePath << std::endl;
                return nullptr;
            }

            isCacheValid = true;
            std::cout << "  成功加载BVH缓存：" << cacheFilePath << std::endl;
            std::cout << "  缓存构建时间：" << outBuildTime << "ms" << std::endl;
            return bvhRoot;
        }
        catch (const std::exception& e) {
            std::cerr << "错误：加载BVH缓存失败 - " << e.what() << std::endl;
            return nullptr;
        }
    }

    // 显式实例化模板函数（避免链接错误）
    template bool SaveBVHCached<AABB>(const std::string&, const std::unique_ptr<BVHNode<AABB>>&, const std::string&, const std::string&, int, int, const std::string&, double);
    template bool SaveBVHCached<BoundingSphere>(const std::string&, const std::unique_ptr<BVHNode<BoundingSphere>>&, const std::string&, const std::string&, int, int, const std::string&, double);
    template std::unique_ptr<BVHNode<AABB>> LoadBVHCached<AABB>(const std::string&, const std::string&, const std::string&, int, int, bool&, double&);
    template std::unique_ptr<BVHNode<BoundingSphere>> LoadBVHCached<BoundingSphere>(const std::string&, const std::string&, const std::string&, int, int, bool&, double&);
    template int CountBVHNodes<AABB>(const std::unique_ptr<BVHNode<AABB>>&);
    template int CountBVHNodes<BoundingSphere>(const std::unique_ptr<BVHNode<BoundingSphere>>&);
    template std::unique_ptr<BVHNode<AABB>> ReconstructBVHNode<AABB>(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>&, int&);
    template std::unique_ptr<BVHNode<BoundingSphere>> ReconstructBVHNode<BoundingSphere>(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>&, int&);
} // namespace BVHAccelerator