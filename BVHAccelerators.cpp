#define _CRT_SECURE_NO_WARNINGS

#include "Core.h"
#include <omp.h>
#include <google/protobuf/repeated_field.h>
#include <atomic>

namespace BVHAccelerator {
    constexpr double kAABBEpsilon = 1e-6;
    constexpr double kEps = 1e-8;
    constexpr double kRayMaxDistance = 1e6;
    constexpr int kMaxLeafSize = 8;
    constexpr int kMaxBVHDepth = 64;
    constexpr int kSAHBucketCount = 16;
    constexpr int kParallelThreshold = 200;
    constexpr int kMaxThreadCount = 8; // 限制最大并行线程数，避免超核心开销

    // 新增：SAH桶结构（存储每个桶的AABB和三角形数量）
    struct SAHBucket {
        AABB bound;
        std::vector<int> triIndices;
        size_t count = 0;
    };

    // SAH核心工具函数（优化：桶排序SAH替换逐点SAH）
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

    // 优化：桶排序SAH分割（替换原FindSAHOptimalSplit）
    bool FindSAHOptimalBucketSplit(const Scenario3D& scene, const std::vector<int>& triIndices, int axis,
        const AABB& parentAABB,
        std::vector<int>& outLeftIndices, std::vector<int>& outRightIndices,
        double& outBestCost) {
        outLeftIndices.clear();
        outRightIndices.clear();
        outBestCost = std::numeric_limits<double>::max();

        // 1. 计算质心范围并初始化桶
        double minCentroid, maxCentroid;
        GetCentroidRange(scene, triIndices, axis, minCentroid, maxCentroid);
        double centroidExtent = maxCentroid - minCentroid;
        if (centroidExtent < kEps) return false;

        SAHBucket buckets[kSAHBucketCount];
        for (int triIdx : triIndices) {
            // 2. 按质心分配到对应桶
            double c = GetTriangleCentroidAxis(scene, triIdx, axis);
            int bucketIdx = static_cast<int>((c - minCentroid) / centroidExtent * kSAHBucketCount);
            bucketIdx = std::clamp(bucketIdx, 0, kSAHBucketCount - 1);
            buckets[bucketIdx].triIndices.push_back(triIdx);
            buckets[bucketIdx].count++;

            // 3. 实时更新桶的AABB
            const auto& tri = scene.triangles[triIdx];
            const auto& p1 = scene.points[tri.p1];
            const auto& p2 = scene.points[tri.p2];
            const auto& p3 = scene.points[tri.p3];
            Point3D triMin(
                std::min({ p1.x, p2.x, p3.x }) - kAABBEpsilon,
                std::min({ p1.y, p2.y, p3.y }) - kAABBEpsilon,
                std::min({ p1.z, p2.z, p3.z }) - kAABBEpsilon
            );
            Point3D triMax(
                std::max({ p1.x, p2.x, p3.x }) + kAABBEpsilon,
                std::max({ p1.y, p2.y, p3.y }) + kAABBEpsilon,
                std::max({ p1.z, p2.z, p3.z }) + kAABBEpsilon
            );
            if (buckets[bucketIdx].count == 1) {
                buckets[bucketIdx].bound = AABB(triMin, triMax);
            }
            else {
                buckets[bucketIdx].bound.min.x = std::min(buckets[bucketIdx].bound.min.x, triMin.x);
                buckets[bucketIdx].bound.min.y = std::min(buckets[bucketIdx].bound.min.y, triMin.y);
                buckets[bucketIdx].bound.min.z = std::min(buckets[bucketIdx].bound.min.z, triMin.z);
                buckets[bucketIdx].bound.max.x = std::max(buckets[bucketIdx].bound.max.x, triMax.x);
                buckets[bucketIdx].bound.max.y = std::max(buckets[bucketIdx].bound.max.y, triMax.y);
                buckets[bucketIdx].bound.max.z = std::max(buckets[bucketIdx].bound.max.z, triMax.z);
            }
        }

        // 4. 预计算前缀和（AABB合并+数量累计）
        std::vector<AABB> leftAABBs(kSAHBucketCount);
        std::vector<size_t> leftCounts(kSAHBucketCount);
        leftAABBs[0] = buckets[0].bound;
        leftCounts[0] = buckets[0].count;
        for (int i = 1; i < kSAHBucketCount; ++i) {
            leftAABBs[i] = AABB::Merge(leftAABBs[i - 1], buckets[i].bound);
            leftCounts[i] = leftCounts[i - 1] + buckets[i].count;
        }

        // 5. 预计算后缀和
        std::vector<AABB> rightAABBs(kSAHBucketCount);
        std::vector<size_t> rightCounts(kSAHBucketCount);
        rightAABBs[kSAHBucketCount - 1] = buckets[kSAHBucketCount - 1].bound;
        rightCounts[kSAHBucketCount - 1] = buckets[kSAHBucketCount - 1].count;
        for (int i = kSAHBucketCount - 2; i >= 0; --i) {
            rightAABBs[i] = AABB::Merge(rightAABBs[i + 1], buckets[i].bound);
            rightCounts[i] = rightCounts[i + 1] + buckets[i].count;
        }

        // 6. 遍历所有可能的分割桶，找最优成本
        int bestSplitBucket = -1;
        for (int splitBucket = 0; splitBucket < kSAHBucketCount - 1; ++splitBucket) {
            if (leftCounts[splitBucket] == 0 || rightCounts[splitBucket + 1] == 0) continue;
            double cost = CalculateSAHCost(parentAABB,
                leftAABBs[splitBucket], leftCounts[splitBucket],
                rightAABBs[splitBucket + 1], rightCounts[splitBucket + 1]);
            if (cost < outBestCost) {
                outBestCost = cost;
                bestSplitBucket = splitBucket;
            }
        }

        // 7. 收集最优分割的左右三角形索引
        if (bestSplitBucket == -1) return false;
        for (int i = 0; i <= bestSplitBucket; ++i) {
            outLeftIndices.insert(outLeftIndices.end(), buckets[i].triIndices.begin(), buckets[i].triIndices.end());
        }
        for (int i = bestSplitBucket + 1; i < kSAHBucketCount; ++i) {
            outRightIndices.insert(outRightIndices.end(), buckets[i].triIndices.begin(), buckets[i].triIndices.end());
        }
        return !outLeftIndices.empty() && !outRightIndices.empty();
    }

    // 优化：适配桶排序SAH的轴选择
    bool SelectSAHOptimalAxis(const Scenario3D& scene, const std::vector<int>& triIndices,
        const AABB& parentAABB,
        std::vector<int>& outLeftIndices, std::vector<int>& outRightIndices) {
        double bestCost = std::numeric_limits<double>::max();
        int bestAxis = -1;
        std::vector<int> bestLeft, bestRight;
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<int> leftIndices, rightIndices;
            double axisCost;
            // 替换为桶排序SAH分割
            if (FindSAHOptimalBucketSplit(scene, triIndices, axis, parentAABB, leftIndices, rightIndices, axisCost)) {
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

    // 优化：Ritter算法+迭代优化（生成更紧凑的包围球）
    BoundingSphere RitterMinSphereOptimized(const Scenario3D& scene, const std::vector<int>& triIndices) {
        if (triIndices.empty()) return BoundingSphere();

        // 步骤1：找初始轴对齐包围盒的对角点
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

        // 步骤2：找初始最远点对
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

        // 步骤3：初始包围球（基于最远点对）
        Point3D center = GeometryUtils::Add(p1, p2);
        center = GeometryUtils::Mul(center, 0.5);
        double radius = GeometryUtils::Distance(p1, center);

        // 步骤4：迭代优化（收缩球心+调整半径，减少冗余体积）
        const int maxIterations = 4;
        for (int iter = 0; iter < maxIterations; ++iter) {
            Point3D farthestPoint = center;
            double farthestDist = 0;
            // 找当前球外最远点
            for (int triIdx : triIndices) {
                const auto& tri = scene.triangles[triIdx];
                const auto& pts = { scene.points[tri.p1], scene.points[tri.p2], scene.points[tri.p3] };
                for (const auto& p : pts) {
                    double dist = GeometryUtils::Distance(center, p);
                    if (dist > radius && dist > farthestDist) {
                        farthestDist = dist;
                        farthestPoint = p;
                    }
                }
            }
            if (farthestDist <= radius) break; // 无球外点，优化结束
            // 收缩球心到包含新点的最小位置
            Point3D dir = GeometryUtils::Sub(farthestPoint, center);
            dir = GeometryUtils::Normalize(dir);
            center = GeometryUtils::Add(center, GeometryUtils::Mul(dir, (farthestDist - radius) * 0.5));
            radius = (radius + farthestDist) * 0.5;
        }

        return BoundingSphere(center, radius + kAABBEpsilon * 0.1); // 减少膨胀系数
    }

    BoundingSphere BuildSphereForTriangles(const Scenario3D& scene, const std::vector<int>& triIndices) {
        if (triIndices.empty()) return BoundingSphere();
        BoundingSphere sphere = RitterMinSphereOptimized(scene, triIndices); // 替换为优化后的包围球生成
        sphere.triangleIndices = triIndices;
        return sphere;
    }

    // 新增：计算节点到射线的最小距离（用于距离优先遍历）
    double CalculateNodeMinDistance(const Point3D& rayOrigin, const Point3D& rayDir, const AABB& aabb) {
        Point3D center = { (aabb.min.x + aabb.max.x) * 0.5,
                           (aabb.min.y + aabb.max.y) * 0.5,
                           (aabb.min.z + aabb.max.z) * 0.5 };
        Point3D oc = GeometryUtils::Sub(center, rayOrigin);
        double proj = GeometryUtils::Dot(oc, rayDir);
        proj = std::max(proj, 0.0);
        Point3D closest = GeometryUtils::Add(rayOrigin, GeometryUtils::Mul(rayDir, proj));
        return GeometryUtils::Distance(closest, center);
    }

    double CalculateNodeMinDistance(const Point3D& rayOrigin, const Point3D& rayDir, const BoundingSphere& sphere) {
        Point3D oc = GeometryUtils::Sub(sphere.center, rayOrigin);
        double proj = GeometryUtils::Dot(oc, rayDir);
        proj = std::max(proj, 0.0);
        Point3D closest = GeometryUtils::Add(rayOrigin, GeometryUtils::Mul(rayDir, proj));
        return GeometryUtils::Distance(closest, sphere.center) - sphere.radius;
    }

    // BVH构建（优化：并行构建+动态叶子阈值）
    std::unique_ptr<BVHNode<AABB>> BuildAABBBVHRecursive(const Scenario3D& scene,
        const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback,
        std::atomic<int>& processedTriangles) {
        auto node = std::make_unique<BVHNode<AABB>>();
        node->bound = BuildAABBForTriangles(scene, triIndices);
        node->depth = depth;

        // 动态叶子阈值：深度越深，阈值越大（减少节点数）
        int dynamicLeafThreshold = kMaxLeafSize + (depth / 8) * 2;
        if (triIndices.size() <= dynamicLeafThreshold || depth >= kMaxBVHDepth) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            processedTriangles += static_cast<int>(triIndices.size());
            return node;
        }

        std::vector<int> leftIndices, rightIndices;
        bool splitSuccess = SelectSAHOptimalAxis(scene, triIndices, node->bound, leftIndices, rightIndices);
        if (!splitSuccess || leftIndices.empty() || rightIndices.empty()) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            processedTriangles += static_cast<int>(triIndices.size());
            return node;
        }

        // 并行构建左右子树（超过阈值才并行，避免线程开销）
        if (triIndices.size() > kParallelThreshold) {
            std::atomic<int> leftProcessed(0), rightProcessed(0);
#pragma omp task shared(node, leftIndices)
            node->left = BuildAABBBVHRecursive(scene, leftIndices, depth + 1, progressCallback, leftProcessed);
#pragma omp task shared(node, rightIndices)
            node->right = BuildAABBBVHRecursive(scene, rightIndices, depth + 1, progressCallback, rightProcessed);
#pragma omp taskwait
            processedTriangles += leftProcessed + rightProcessed;
        }
        else {
            node->left = BuildAABBBVHRecursive(scene, leftIndices, depth + 1, progressCallback, processedTriangles);
            node->right = BuildAABBBVHRecursive(scene, rightIndices, depth + 1, progressCallback, processedTriangles);
        }

        return node;
    }

    std::unique_ptr<BVHNode<AABB>> BuildAABBBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback) {
        std::vector<int> triIndices(scene.triangles.size());
        std::iota(triIndices.begin(), triIndices.end(), 0);
        std::atomic<int> processedTriangles(0);
        omp_set_num_threads(std::min(kMaxThreadCount, static_cast<int>(std::thread::hardware_concurrency())));
#pragma omp parallel
#pragma omp single nowait
        {
            return BuildAABBBVHRecursive(scene, triIndices, 0, progressCallback, processedTriangles);
        }
    }

    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVHRecursive(const Scenario3D& scene,
        const std::vector<int>& triIndices, int depth, const BVHProgressCallback& progressCallback,
        std::atomic<int>& processedTriangles) {
        auto node = std::make_unique<BVHNode<BoundingSphere>>();
        node->bound = BuildSphereForTriangles(scene, triIndices);
        node->depth = depth;

        // 动态叶子阈值
        int dynamicLeafThreshold = kMaxLeafSize + (depth / 8) * 2;
        if (triIndices.size() <= dynamicLeafThreshold || depth >= kMaxBVHDepth) {
            node->isLeaf = true;
            if (progressCallback) progressCallback(static_cast<int>(triIndices.size()));
            processedTriangles += static_cast<int>(triIndices.size());
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
            processedTriangles += static_cast<int>(triIndices.size());
            return node;
        }

        // 并行构建左右子树
        if (triIndices.size() > kParallelThreshold) {
            std::atomic<int> leftProcessed(0), rightProcessed(0);
#pragma omp task shared(node, leftIndices)
            node->left = BuildSphereBVHRecursive(scene, leftIndices, depth + 1, progressCallback, leftProcessed);
#pragma omp task shared(node, rightIndices)
            node->right = BuildSphereBVHRecursive(scene, rightIndices, depth + 1, progressCallback, rightProcessed);
#pragma omp taskwait
            processedTriangles += leftProcessed + rightProcessed;
        }
        else {
            node->left = BuildSphereBVHRecursive(scene, leftIndices, depth + 1, progressCallback, processedTriangles);
            node->right = BuildSphereBVHRecursive(scene, rightIndices, depth + 1, progressCallback, processedTriangles);
        }

        return node;
    }

    std::unique_ptr<BVHNode<BoundingSphere>> BuildSphereBVH(const Scenario3D& scene, const BVHProgressCallback& progressCallback) {
        std::vector<int> triIndices(scene.triangles.size());
        std::iota(triIndices.begin(), triIndices.end(), 0);
        std::atomic<int> processedTriangles(0);
        omp_set_num_threads(std::min(kMaxThreadCount, static_cast<int>(std::thread::hardware_concurrency())));
#pragma omp parallel
#pragma omp single nowait
        {
            return BuildSphereBVHRecursive(scene, triIndices, 0, progressCallback, processedTriangles);
        }
    }

    // 射线检测（优化：相交算法+距离优先遍历+早期终止+结果收集优化）
    bool RayIntersectAABB_Box(const Point3D& rayOrigin, const Point3D& rayDir, const Point3D& rayDirInv, const AABB& aabb) {
        // 预计算逆向量，减少除法（核心优化）
        double tMinX = (aabb.min.x - rayOrigin.x) * rayDirInv.x;
        double tMaxX = (aabb.max.x - rayOrigin.x) * rayDirInv.x;
        if (tMinX > tMaxX) std::swap(tMinX, tMaxX);

        double tMinY = (aabb.min.y - rayOrigin.y) * rayDirInv.y;
        double tMaxY = (aabb.max.y - rayOrigin.y) * rayDirInv.y;
        if (tMinY > tMaxY) std::swap(tMinY, tMaxY);

        if (tMinX > tMaxY || tMinY > tMaxX) return false;
        tMinX = std::max(tMinX, tMinY);
        tMaxX = std::min(tMaxX, tMaxY);

        double tMinZ = (aabb.min.z - rayOrigin.z) * rayDirInv.z;
        double tMaxZ = (aabb.max.z - rayOrigin.z) * rayDirInv.z;
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
            return distSq <= sphere.radius * sphere.radius + kEps; // 半径平方，避免sqrt
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
        const Point3D& rayOrigin, const Point3D& rayDir, double& tMin) {
        SingleHitResult hit;
        const auto& tri = scene.triangles[triIndex];
        const auto& p1 = scene.points[tri.p1];
        const auto& p2 = scene.points[tri.p2];
        const auto& p3 = scene.points[tri.p3];

        // 早期终止：如果三角形中心到射线起点距离已大于当前tMin，直接跳过
        Point3D triCenter = GeometryUtils::Add(p1, GeometryUtils::Add(p2, p3));
        triCenter = GeometryUtils::Mul(triCenter, 1.0 / 3.0);
        double centerDist = GeometryUtils::Distance(rayOrigin, triCenter);
        if (centerDist > tMin + kEps) return hit;

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
        if (t > kEps && t < kRayMaxDistance && t < tMin) {
            hit.hasHit = true;
            hit.distance = t;
            hit.triangleIndex = triIndex;
            hit.hitPoint = GeometryUtils::Add(rayOrigin, GeometryUtils::Mul(rayDir, t));
            tMin = t; // 更新全局最近距离
        }
        return hit;
    }

    // 优化：AABB-BVH距离优先遍历+早期终止
    void TraverseAABBBVH(const Scenario3D& scene, const BVHNode<AABB>* node,
        const Point3D& rayOrigin, const Point3D& rayDir, const Point3D& rayDirInv,
        RayIntersectResult& result, std::atomic<double>& tMin,
        const RayProgressCallback& progressCallback, std::mutex& progressMtx) {
        if (!node) return;

        // 早期终止：节点最小距离 >= 当前最近碰撞距离，跳过该节点及其子树
        double nodeMinDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->bound);
        if (nodeMinDist >= tMin - kEps) return;

        // 包围盒相交检测（使用预计算的逆向量）
        if (!RayIntersectAABB_Box(rayOrigin, rayDir, rayDirInv, node->bound)) return;

        if (node->isLeaf) {
            const auto& triIndices = node->bound.triangleIndices;
            const int triCount = static_cast<int>(triIndices.size());
            thread_local std::vector<SingleHitResult> localHits;
            localHits.clear();

            if (triCount > kParallelThreshold) {
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
                for (int i = 0; i < triCount; ++i) {
                    int triIdx = triIndices[i];
                    double localTMin = tMin.load(std::memory_order_relaxed);
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir, localTMin);
                    if (hit.hasHit) {
                        localHits.push_back(hit);
                        // 原子更新全局tMin
                        double expected = tMin.load(std::memory_order_relaxed);
                        while (localTMin < expected && !tMin.compare_exchange_weak(expected, localTMin,
                            std::memory_order_release, std::memory_order_relaxed));
                    }
                    if (progressCallback) {
                        std::lock_guard<std::mutex> lock(progressMtx);
                        progressCallback(1);
                    }
                }
            }
            else {
                for (int triIdx : triIndices) {
                    double localTMin = tMin.load(std::memory_order_relaxed);
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir, localTMin);
                    if (hit.hasHit) {
                        localHits.push_back(hit);
                        double expected = tMin.load(std::memory_order_relaxed);
                        while (localTMin < expected && !tMin.compare_exchange_weak(expected, localTMin,
                            std::memory_order_release, std::memory_order_relaxed));
                    }
                    if (progressCallback) progressCallback(1);
                }
            }

            // 批量合并结果（避免多次排序）
            if (!localHits.empty()) {
                std::sort(localHits.begin(), localHits.end(), [](const SingleHitResult& a, const SingleHitResult& b) {
                    return a.distance < b.distance;
                    });
                result.MergeLocalHits(localHits);
            }
            return;
        }

        // 距离优先遍历：优先遍历更近的子节点
        double leftDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->left->bound);
        double rightDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->right->bound);
        if (leftDist < rightDist) {
            TraverseAABBBVH(scene, node->left.get(), rayOrigin, rayDir, rayDirInv, result, tMin, progressCallback, progressMtx);
            TraverseAABBBVH(scene, node->right.get(), rayOrigin, rayDir, rayDirInv, result, tMin, progressCallback, progressMtx);
        }
        else {
            TraverseAABBBVH(scene, node->right.get(), rayOrigin, rayDir, rayDirInv, result, tMin, progressCallback, progressMtx);
            TraverseAABBBVH(scene, node->left.get(), rayOrigin, rayDir, rayDirInv, result, tMin, progressCallback, progressMtx);
        }
    }

    // 优化：Sphere-BVH距离优先遍历+早期终止
    void TraverseSphereBVH(const Scenario3D& scene, const BVHNode<BoundingSphere>* node,
        const Point3D& rayOrigin, const Point3D& rayDir,
        RayIntersectResult& result, std::atomic<double>& tMin,
        const RayProgressCallback& progressCallback, std::mutex& progressMtx) {
        if (!node) return;

        double nodeMinDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->bound);
        if (nodeMinDist >= tMin - kEps) return;

        if (!RayIntersectSphere(rayOrigin, rayDir, node->bound)) return;

        if (node->isLeaf) {
            const auto& triIndices = node->bound.triangleIndices;
            const int triCount = static_cast<int>(triIndices.size());
            thread_local std::vector<SingleHitResult> localHits;
            localHits.clear();

            if (triCount > kParallelThreshold) {
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
                for (int i = 0; i < triCount; ++i) {
                    int triIdx = triIndices[i];
                    double localTMin = tMin.load(std::memory_order_relaxed);
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir, localTMin);
                    if (hit.hasHit) {
                        localHits.push_back(hit);
                        double expected = tMin.load(std::memory_order_relaxed);
                        while (localTMin < expected && !tMin.compare_exchange_weak(expected, localTMin,
                            std::memory_order_release, std::memory_order_relaxed));
                    }
                    if (progressCallback) {
                        std::lock_guard<std::mutex> lock(progressMtx);
                        progressCallback(1);
                    }
                }
            }
            else {
                for (int triIdx : triIndices) {
                    double localTMin = tMin.load(std::memory_order_relaxed);
                    SingleHitResult hit = RayIntersectTriangle(scene, triIdx, rayOrigin, rayDir, localTMin);
                    if (hit.hasHit) {
                        localHits.push_back(hit);
                        double expected = tMin.load(std::memory_order_relaxed);
                        while (localTMin < expected && !tMin.compare_exchange_weak(expected, localTMin,
                            std::memory_order_release, std::memory_order_relaxed));
                    }
                    if (progressCallback) progressCallback(1);
                }
            }

            if (!localHits.empty()) {
                std::sort(localHits.begin(), localHits.end(), [](const SingleHitResult& a, const SingleHitResult& b) {
                    return a.distance < b.distance;
                    });
                result.MergeLocalHits(localHits);
            }
            return;
        }

        // 距离优先遍历
        double leftDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->left->bound);
        double rightDist = CalculateNodeMinDistance(rayOrigin, rayDir, node->right->bound);
        if (leftDist < rightDist) {
            TraverseSphereBVH(scene, node->left.get(), rayOrigin, rayDir, result, tMin, progressCallback, progressMtx);
            TraverseSphereBVH(scene, node->right.get(), rayOrigin, rayDir, result, tMin, progressCallback, progressMtx);
        }
        else {
            TraverseSphereBVH(scene, node->right.get(), rayOrigin, rayDir, result, tMin, progressCallback, progressMtx);
            TraverseSphereBVH(scene, node->left.get(), rayOrigin, rayDir, result, tMin, progressCallback, progressMtx);
        }
    }

    RayIntersectResult RayIntersectAABB(const Scenario3D& scene, const BVHNode<AABB>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback) {
        RayIntersectResult result;
        std::mutex progressMtx;
        std::atomic<double> tMin(kRayMaxDistance); // 全局最近碰撞距离（原子变量线程安全）

        // 预计算射线方向逆向量（仅计算一次）
        Point3D rayDirInv(
            rayDir.x == 0 ? (rayOrigin.x > 0 ? kRayMaxDistance : -kRayMaxDistance) : 1.0 / rayDir.x,
            rayDir.y == 0 ? (rayOrigin.y > 0 ? kRayMaxDistance : -kRayMaxDistance) : 1.0 / rayDir.y,
            rayDir.z == 0 ? (rayOrigin.z > 0 ? kRayMaxDistance : -kRayMaxDistance) : 1.0 / rayDir.z
        );

        TraverseAABBBVH(scene, root, rayOrigin, rayDir, rayDirInv, result, tMin, progressCallback, progressMtx);
        return result;
    }

    RayIntersectResult RayIntersectSphere(const Scenario3D& scene, const BVHNode<BoundingSphere>* root, const Point3D& rayOrigin, const Point3D& rayDir, const RayProgressCallback& progressCallback) {
        RayIntersectResult result;
        std::mutex progressMtx;
        std::atomic<double> tMin(kRayMaxDistance);

        TraverseSphereBVH(scene, root, rayOrigin, rayDir, result, tMin, progressCallback, progressMtx);
        return result;
    }

    // Protobuf导出（保持不变）
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

    // BVH节点数统计（保持不变）
    template <typename BoundType>
    int CountBVHNodes(const std::unique_ptr<BVHNode<BoundType>>& root) {
        if (!root) return 0;
        return 1 + CountBVHNodes(root->left) + CountBVHNodes(root->right);
    }

    // BVH缓存保存/加载（保持不变）
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
            cacheMeta->set_optimize_method("SAH-Optimized(Bucket)"); // 更新优化方式说明

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

    // 显式实例化模板函数（保持不变）
    template bool SaveBVHCached<AABB>(const std::string&, const std::unique_ptr<BVHNode<AABB>>&, const std::string&, const std::string&, int, int, const std::string&, double);
    template bool SaveBVHCached<BoundingSphere>(const std::string&, const std::unique_ptr<BVHNode<BoundingSphere>>&, const std::string&, const std::string&, int, int, const std::string&, double);
    template std::unique_ptr<BVHNode<AABB>> LoadBVHCached<AABB>(const std::string&, const std::string&, const std::string&, int, int, bool&, double&);
    template std::unique_ptr<BVHNode<BoundingSphere>> LoadBVHCached<BoundingSphere>(const std::string&, const std::string&, const std::string&, int, int, bool&, double&);
    template int CountBVHNodes<AABB>(const std::unique_ptr<BVHNode<AABB>>&);
    template int CountBVHNodes<BoundingSphere>(const std::unique_ptr<BVHNode<BoundingSphere>>&);
    template std::unique_ptr<BVHNode<AABB>> ReconstructBVHNode<AABB>(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>&, int&);
    template std::unique_ptr<BVHNode<BoundingSphere>> ReconstructBVHNode<BoundingSphere>(const ::google::protobuf::RepeatedPtrField<BVHData::BVHNodeProto>&, int&);
} // namespace BVHAccelerator