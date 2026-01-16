#define _CRT_SECURE_NO_WARNINGS
#include "Core.h"
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // ===================== 1. 目录管理：按场景分类缓存结构 =====================
    const std::string cacheRootDir = "Cache";
    const std::string modelFilePath = "obj\\m.obj";
    const fs::path modelFile(modelFilePath);
    const std::string sceneName = modelFile.stem().string();
    const std::string sceneCacheDir = cacheRootDir + "\\" + sceneName;

    // 递归创建场景缓存目录
    if (!CreateDirectoryRecursive(sceneCacheDir)) {
        std::cerr << "错误：创建场景缓存目录失败，程序退出" << std::endl;
        return -1;
    }

    // ===================== 2. 日志初始化：控制台+文件双输出 =====================
    const std::string logFilePath = sceneCacheDir + "\\log.txt";
    std::ofstream logFile(logFilePath);
    if (!logFile.is_open()) {
        std::cerr << "错误：无法打开日志文件 " << logFilePath << std::endl;
        return -1;
    }

    // 保存控制台原始缓冲区
    std::streambuf* originalCoutBuf = std::cout.rdbuf();
    std::streambuf* originalCerrBuf = std::cerr.rdbuf();
    // 重定向cout/cerr到日志文件（进度条仅在控制台显示，需临时切换）
    std::cout.rdbuf(logFile.rdbuf());
    std::cerr.rdbuf(logFile.rdbuf());

    // ===================== 3. 配置参数与模型加载（带进度条） =====================
    // 射线参数
    Point3D rayOrigin(4000.0, 500, 3000);
    Point3D rayDir(1.0, 0.0, 1.0);
    rayDir = GeometryUtils::Normalize(rayDir);

    // 加载模型（带进度条）
    Scenario3D scene;
    bool loadSuccess = false;
    const std::string modelExt = modelFile.extension().string();

    std::cout.rdbuf(originalCoutBuf); // 切换到控制台显示进度条
    std::cout << "============================================" << std::endl;
    std::cout << "            模型加载中...                    " << std::endl;
    std::cout.rdbuf(logFile.rdbuf()); // 切回日志文件

    if (modelExt == ".obj" || modelExt == ".OBJ") {
        // 统计OBJ文件行数，作为进度条总步数
        int objTotalLines = CountFileLines(modelFilePath);
        ProgressBar loadProgress(objTotalLines, "加载场景");

        std::cout.rdbuf(originalCoutBuf); // 控制台显示进度条
        loadSuccess = ModelImporter::LoadOBJ(modelFilePath, scene, &loadProgress);
        std::cout.rdbuf(logFile.rdbuf()); // 切回日志文件
    }
    else {
        std::cerr << "错误：不支持的模型格式 " << modelExt << "，仅支持OBJ格式" << std::endl;
    }

    if (!loadSuccess) {
        std::cerr << "模型加载失败，程序退出" << std::endl;
        // 恢复控制台并清理资源
        std::cout.rdbuf(originalCoutBuf);
        std::cerr.rdbuf(originalCerrBuf);
        logFile.close();
        std::cout << "错误：模型加载失败！" << std::endl;
        return -1;
    }

    // 输出模型加载信息（控制台+日志）
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "============================================" << std::endl;
    std::cout << "            模型加载成功                    " << std::endl;
    std::cout << "  模型文件：" << modelFilePath << std::endl;
    std::cout << "  - 顶点数量：" << scene.points.size() << std::endl;
    std::cout << "  - 三角形数量：" << scene.triangles.size() << std::endl;
    std::cout.rdbuf(logFile.rdbuf());

    // ===================== 4. BVH构建/加载（带进度条） =====================
    const std::string objFileMTime = GetFileLastModifyTime(modelFilePath);
    const std::string aabbCachePath = sceneCacheDir + "\\aabb_bvh_cache.pb";
    const std::string sphereCachePath = sceneCacheDir + "\\sphere_bvh_cache.pb";

    std::unique_ptr<BVHNode<AABB>> aabbBVH;
    std::unique_ptr<BVHNode<BoundingSphere>> sphereBVH;
    double aabbInitTime = 0.0, sphereInitTime = 0.0;
    bool isAABBCacheValid = false, isSphereCacheValid = false;

    // 切换到控制台显示BVH进度
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "\n============================================" << std::endl;
    std::cout << "            BVH树构建/加载中...                  " << std::endl;
    std::cout.rdbuf(logFile.rdbuf());

    // 加载AABB-BVH（带进度条）
    std::cout.rdbuf(originalCoutBuf);
    aabbBVH = BVHAccelerator::LoadBVHCached<AABB>(
        aabbCachePath, modelFilePath, objFileMTime,
        static_cast<int>(scene.points.size()), static_cast<int>(scene.triangles.size()),
        isAABBCacheValid, aabbInitTime
    );
    if (!isAABBCacheValid) {
        // 缓存无效，重新构建（带进度条）
        ProgressBar aabbBuildProgress(static_cast<int>(scene.triangles.size()), "构建AABB-BVH");
        const TimePoint aabbInitStart = GetCurrentTime();
        aabbBVH = BVHAccelerator::BuildAABBBVH(scene, [&](int steps) {
            aabbBuildProgress.Update(steps);
            });
        const TimePoint aabbInitEnd = GetCurrentTime();
        aabbInitTime = CalculateElapsedMs(aabbInitStart, aabbInitEnd);
        aabbBuildProgress.Finish();

        // 保存新缓存
        if (!objFileMTime.empty()) {
            BVHAccelerator::SaveBVHCached<AABB>(
                aabbCachePath, aabbBVH, modelFilePath, objFileMTime,
                static_cast<int>(scene.points.size()), static_cast<int>(scene.triangles.size()),
                "AABB-BVH(SAH-Optimized)", aabbInitTime
            );
        }
    }
    else {
        // 缓存有效，直接显示加载完成
        std::cout << "构建AABB-BVH [##################################] 100% (加载缓存)" << std::endl;
    }
    std::cout.rdbuf(logFile.rdbuf());

    // 加载Sphere-BVH（带进度条）
    std::cout.rdbuf(originalCoutBuf);
    sphereBVH = BVHAccelerator::LoadBVHCached<BoundingSphere>(
        sphereCachePath, modelFilePath, objFileMTime,
        static_cast<int>(scene.points.size()), static_cast<int>(scene.triangles.size()),
        isSphereCacheValid, sphereInitTime
    );
    if (!isSphereCacheValid) {
        // 缓存无效，重新构建（带进度条）
        ProgressBar sphereBuildProgress(static_cast<int>(scene.triangles.size()), "构建Sphere-BVH");
        const TimePoint sphereInitStart = GetCurrentTime();
        sphereBVH = BVHAccelerator::BuildSphereBVH(scene, [&](int steps) {
            sphereBuildProgress.Update(steps);
            });
        const TimePoint sphereInitEnd = GetCurrentTime();
        sphereInitTime = CalculateElapsedMs(sphereInitStart, sphereInitEnd);
        sphereBuildProgress.Finish();

        // 保存新缓存
        if (!objFileMTime.empty()) {
            BVHAccelerator::SaveBVHCached<BoundingSphere>(
                sphereCachePath, sphereBVH, modelFilePath, objFileMTime,
                static_cast<int>(scene.points.size()), static_cast<int>(scene.triangles.size()),
                "Sphere-BVH(SAH-Optimized)", sphereInitTime
            );
        }
    }
    else {
        // 缓存有效，直接显示加载完成
        std::cout << "构建Sphere-BVH [##################################] 100% (加载缓存)" << std::endl;
    }
    std::cout.rdbuf(logFile.rdbuf());

    // 输出BVH构建/加载结果
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "\n============================================" << std::endl;
    std::cout << "  AABB-BVH" << (isAABBCacheValid ? "加载" : "构建") << "完成！耗时：" << std::fixed << std::setprecision(3) << aabbInitTime << "ms" << std::endl;
    std::cout << "  Sphere-BVH" << (isSphereCacheValid ? "加载" : "构建") << "完成！耗时：" << std::fixed << std::setprecision(3) << sphereInitTime << "ms" << std::endl;
    std::cout.rdbuf(logFile.rdbuf());

    // ===================== 5. 射线碰撞检测（带进度条） =====================
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "\n============================================" << std::endl;
    std::cout << "          射线碰撞检测中...                 " << std::endl;
    std::cout.rdbuf(logFile.rdbuf());

    // AABB-BVH检测（带进度条）
    std::cout.rdbuf(originalCoutBuf);
    ProgressBar aabbDetectProgress(static_cast<int>(scene.triangles.size()), "AABB-BVH检测");
    const TimePoint aabbDetectStart = GetCurrentTime();
    const RayIntersectResult aabbResult = BVHAccelerator::RayIntersectAABB(
        scene, aabbBVH.get(), rayOrigin, rayDir, [&](int steps) {
            aabbDetectProgress.Update(steps);
        }
    );
    const TimePoint aabbDetectEnd = GetCurrentTime();
    const double aabbDetectTime = CalculateElapsedMs(aabbDetectStart, aabbDetectEnd);
    aabbDetectProgress.Finish();

    // Sphere-BVH检测（带进度条）
    ProgressBar sphereDetectProgress(static_cast<int>(scene.triangles.size()), "Sphere-BVH检测");
    const TimePoint sphereDetectStart = GetCurrentTime();
    const RayIntersectResult sphereResult = BVHAccelerator::RayIntersectSphere(
        scene, sphereBVH.get(), rayOrigin, rayDir, [&](int steps) {
            sphereDetectProgress.Update(steps);
        }
    );
    const TimePoint sphereDetectEnd = GetCurrentTime();
    const double sphereDetectTime = CalculateElapsedMs(sphereDetectStart, sphereDetectEnd);
    sphereDetectProgress.Finish();
    std::cout.rdbuf(logFile.rdbuf());

    // 输出检测结果
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "\n============================================" << std::endl;
    std::cout << "  AABB-BVH检测完成！耗时：" << std::fixed << std::setprecision(3) << aabbDetectTime << "ms" << std::endl;
    std::cout << "  Sphere-BVH检测完成！耗时：" << std::fixed << std::setprecision(3) << sphereDetectTime << "ms" << std::endl;
    std::cout.rdbuf(logFile.rdbuf());

    // ===================== 6. 差异日志与Protobuf导出 =====================
    // 统计BVH节点数
    const int aabbNodeCount = BVHAccelerator::CountBVHNodes(aabbBVH);
    const int sphereNodeCount = BVHAccelerator::CountBVHNodes(sphereBVH);
    const int nodeCountDiff = aabbNodeCount - sphereNodeCount;
    const double initTimeDiff = aabbInitTime - sphereInitTime;
    const double detectTimeDiff = aabbDetectTime - sphereDetectTime;

    // 输出差异对比日志
    std::cout << "\n============================================" << std::endl;
    std::cout << "            AABB-BVH vs Sphere-BVH 差异对比             " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "【构建耗时差异】" << std::endl;
    std::cout << "  AABB-BVH " << (isAABBCacheValid ? "(复用)" : "") << "耗时：" << std::fixed << std::setprecision(3) << aabbInitTime << "ms" << std::endl;
    std::cout << "  Sphere-BVH " << (isSphereCacheValid ? "(复用)" : "") << "耗时：" << std::fixed << std::setprecision(3) << sphereInitTime << "ms" << std::endl;
    std::cout << "  差异（AABB - Sphere）：" << std::fixed << std::setprecision(3) << initTimeDiff << "ms" << std::endl;
    std::cout << "  结论：" << (initTimeDiff > 0 ? "Sphere-BVH构建更快" : "AABB-BVH构建更快") << std::endl;

    std::cout << "\n【检测耗时差异】" << std::endl;
    std::cout << "  AABB-BVH 耗时：" << std::fixed << std::setprecision(3) << aabbDetectTime << "ms" << std::endl;
    std::cout << "  Sphere-BVH 耗时：" << std::fixed << std::setprecision(3) << sphereDetectTime << "ms" << std::endl;
    std::cout << "  差异（AABB - Sphere）：" << std::fixed << std::setprecision(3) << detectTimeDiff << "ms" << std::endl;
    std::cout << "  结论：" << (detectTimeDiff > 0 ? "Sphere-BVH检测更快" : "AABB-BVH检测更快") << std::endl;

    std::cout << "\n【节点数差异】" << std::endl;
    std::cout << "  AABB-BVH 节点总数：" << aabbNodeCount << std::endl;
    std::cout << "  Sphere-BVH 节点总数：" << sphereNodeCount << std::endl;
    std::cout << "  差异（AABB - Sphere）：" << nodeCountDiff << "个" << std::endl;
    std::cout << "============================================" << std::endl;

    // Protobuf导出（输出到场景缓存目录）
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "\n============================================" << std::endl;
    std::cout << "          结果导出中...                     " << std::endl;
    Visualizer::ExportRayAndHitToProtobuf(sceneCacheDir + "\\ray_hit_data.pb", rayOrigin, rayDir, aabbResult);
    BVHAccelerator::ExportBVHToProtobuf(sceneCacheDir + "\\aabb_bvh_structure.pb", aabbBVH.get(), "AABB-BVH(SAH-Optimized)");
    BVHAccelerator::ExportBVHToProtobuf(sceneCacheDir + "\\sphere_bvh_structure.pb", sphereBVH.get(), "Sphere-BVH(SAH-Optimized)");
    std::cout.rdbuf(logFile.rdbuf());

    // ===================== 7. 最终结果输出与资源清理 =====================
    std::cout.rdbuf(originalCoutBuf);
    std::cout << "============================================" << std::endl;
    std::cout << "            程序执行完成！                  " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  输出文件清单（场景缓存目录：" << sceneCacheDir << "）：" << std::endl;
    std::cout << "  1. 详细日志：log.txt" << std::endl;
    std::cout << "  2. 射线碰撞数据：ray_hit_data.pb" << std::endl;
    std::cout << "  3. AABB-BVH结构：aabb_bvh_structure.pb" << std::endl;
    std::cout << "  4. Sphere-BVH结构：sphere_bvh_structure.pb" << std::endl;
    std::cout << "  5. AABB-BVH缓存：aabb_bvh_cache.pb" << (isAABBCacheValid ? "（已复用）" : "（新生成）") << std::endl;
    std::cout << "  6. Sphere-BVH缓存：sphere_bvh_cache.pb" << (isSphereCacheValid ? "（已复用）" : "（新生成）") << std::endl;

    // 恢复控制台并关闭日志文件
    std::cout.rdbuf(originalCoutBuf);
    std::cerr.rdbuf(originalCerrBuf);
    logFile.close();

    return 0;
}