#define _CRT_SECURE_NO_WARNINGS
#include "Core.h"
#include <windows.h>


// ProgressBar 构造函数实现（与声明一致）
ProgressBar::ProgressBar(int totalSteps, const std::string& title, int updateInterval)
    : total_steps_(totalSteps), current_steps_(0), title_(title),
    is_finished_(false), update_interval_(updateInterval) {}

void ProgressBar::Update(int steps) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_finished_) return;
    current_steps_ = min(current_steps_ + steps, total_steps_);
    // 仅当达到更新间隔或进度100%时才显示
    if (current_steps_ % update_interval_ == 0 || current_steps_ == total_steps_) {
        Display();
    }
}

void ProgressBar::Finish() {
    std::lock_guard<std::mutex> lock(mtx_);
    current_steps_ = total_steps_;
    is_finished_ = true;
    Display();
    std::cout << std::endl;
}

void ProgressBar::Display() {
    const int barWidth = 50;
    double progress = static_cast<double>(current_steps_) / total_steps_;
    int percent = static_cast<int>(progress * 100);
    int filledWidth = static_cast<int>(progress * barWidth);

    std::string bar(filledWidth, '#');
    bar += std::string(barWidth - filledWidth, ' ');

    std::cout << "\r" << title_ << " [" << bar << "] " << std::setw(3) << percent << "% "
        << "(" << current_steps_ << "/" << total_steps_ << ")";
}

// 全局工具函数实现
std::string GetRuntimeTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timeT);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

std::string GetFileLastModifyTime(const std::string& filePath) {
    HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return "";
    FILETIME ftWrite;
    if (!GetFileTime(hFile, NULL, NULL, &ftWrite)) {
        CloseHandle(hFile);
        return "";
    }
    CloseHandle(hFile);
    SYSTEMTIME st;
    FileTimeToSystemTime(&ftWrite, &st);
    char buf[64];
    sprintf_s(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return std::string(buf);
}

bool CreateDirectoryRecursive(const std::string& dirPath) {
    try {
        std::filesystem::create_directories(dirPath);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "错误：创建目录失败 " << dirPath << " - " << e.what() << std::endl;
        return false;
    }
}

int CountFileLines(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) return 1;
    int lines = 0;
    std::string line;
    while (std::getline(file, line)) lines++;
    file.close();
    return lines;
}

// 几何工具函数实现
namespace GeometryUtils {
    double Dot(const Point3D& a, const Point3D& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Point3D Cross(const Point3D& a, const Point3D& b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    double Distance(const Point3D& a, const Point3D& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    Point3D Normalize(const Point3D& a) {
        double len = Distance(a, Point3D(0, 0, 0));
        if (len < kEps) return a;
        return { a.x / len, a.y / len, a.z / len };
    }

    Point3D Sub(const Point3D& a, const Point3D& b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    Point3D Add(const Point3D& a, const Point3D& b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    Point3D Mul(const Point3D& a, double d) {
        return { a.x * d, a.y * d, a.z * d };
    }

    std::vector<Point3D> GenerateRayPoints(const Point3D& origin, const Point3D& dir, double maxDist, int sampleCount) {
        std::vector<Point3D> points;
        double step = maxDist / sampleCount;
        for (int i = 0; i <= sampleCount; ++i) {
            double t = step * i;
            points.emplace_back(
                origin.x + dir.x * t,
                origin.y + dir.y * t,
                origin.z + dir.z * t
            );
        }
        return points;
    }
}

// 可视化工具实现
namespace Visualizer {
    void ShowRayAndHitDetail(const Point3D& rayOrigin, const Point3D& rayDir, const RayIntersectResult& result) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "            射线+碰撞点详细信息             " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "【射线参数】" << std::endl;
        std::cout << "起点坐标：(" << rayOrigin.x << ", " << rayOrigin.y << ", " << rayOrigin.z << ")" << std::endl;
        std::cout << "方向向量：(" << rayDir.x << ", " << rayDir.y << ", " << rayDir.z << ")（已归一化）" << std::endl;

        SingleHitResult closestHit;
        std::vector<Point3D> rayPoints;
        if (result.hasAnyHit) {
            std::cout << "\n【所有碰撞结果（按距离排序）】" << std::endl;
            for (size_t i = 0; i < result.allHits.size(); ++i) {
                const auto& hit = result.allHits[i];
                std::cout << "第" << i + 1 << "个碰撞：" << std::endl;
                std::cout << "  碰撞点坐标：(" << hit.hitPoint.x << ", " << hit.hitPoint.y << ", " << hit.hitPoint.z << ")" << std::endl;
                std::cout << "  射线距离 t：" << hit.distance << std::endl;
                std::cout << "  碰撞三角形索引：" << hit.triangleIndex << std::endl;
            }
            closestHit = result.GetClosestHit();
            rayPoints = GeometryUtils::GenerateRayPoints(rayOrigin, rayDir, closestHit.distance);
            std::cout << "\n【射线轨迹采样点（前5个+最近碰撞点）】" << std::endl;
            for (int i = 0; i < min(5, (int)rayPoints.size()); ++i) {
                std::cout << "t=" << (closestHit.distance / rayPoints.size()) * i << ": ("
                    << rayPoints[i].x << ", " << rayPoints[i].y << ", " << rayPoints[i].z << ")" << std::endl;
            }
            std::cout << "t=" << closestHit.distance << ": ("
                << rayPoints.back().x << ", " << rayPoints.back().y << ", " << rayPoints.back().z << ") 【最近碰撞点】" << std::endl;
        }
        else {
            std::cout << "\n【无碰撞点】" << std::endl;
            rayPoints = GeometryUtils::GenerateRayPoints(rayOrigin, rayDir, kRayMaxDistance, 10);
            std::cout << "射线最大距离采样点（t=" << kRayMaxDistance << "）：("
                << rayPoints.back().x << ", " << rayPoints.back().y << ", " << rayPoints.back().z << ")" << std::endl;
        }
    }

    void ExportRayAndHitToProtobuf(const std::string& filePath, const Point3D& rayOrigin, const Point3D& rayDir, const RayIntersectResult& result) {
        try {
            BVHData::RayHitDataProto rayData;
            auto* metadata = rayData.mutable_metadata();
            metadata->set_export_time(GetRuntimeTimestamp());
            metadata->set_ray_max_distance(ClampValidNumber(kRayMaxDistance));
            metadata->set_has_collision(result.hasAnyHit);
            metadata->set_collision_count(static_cast<int>(result.allHits.size()));

            auto* rayProto = rayData.mutable_ray();
            auto* originProto = rayProto->mutable_origin();
            originProto->set_x(ClampValidNumber(rayOrigin.x));
            originProto->set_y(ClampValidNumber(rayOrigin.y));
            originProto->set_z(ClampValidNumber(rayOrigin.z));
            auto* dirProto = rayProto->mutable_direction();
            dirProto->set_x(ClampValidNumber(rayDir.x));
            dirProto->set_y(ClampValidNumber(rayDir.y));
            dirProto->set_z(ClampValidNumber(rayDir.z));
            rayProto->set_normalized(true);

            double maxDist = result.hasAnyHit ? result.GetClosestHit().distance : kRayMaxDistance;
            maxDist = ClampValidNumber(maxDist, kRayMaxDistance);
            std::vector<Point3D> rayPoints = GeometryUtils::GenerateRayPoints(rayOrigin, rayDir, maxDist, 50);
            for (size_t i = 0; i < rayPoints.size(); ++i) {
                double t = (maxDist / rayPoints.size()) * i;
                auto* trajPoint = rayData.add_ray_trajectory();
                trajPoint->set_index(static_cast<int>(i));
                trajPoint->set_t(t);
                auto* pointProto = trajPoint->mutable_point();
                pointProto->set_x(ClampValidNumber(rayPoints[i].x));
                pointProto->set_y(ClampValidNumber(rayPoints[i].y));
                pointProto->set_z(ClampValidNumber(rayPoints[i].z));
            }

            if (result.hasAnyHit) {
                for (size_t i = 0; i < result.allHits.size(); ++i) {
                    const auto& hit = result.allHits[i];
                    auto* collisionProto = rayData.add_collisions();
                    collisionProto->set_index(static_cast<int>(i));
                    collisionProto->set_triangle_index(hit.triangleIndex);
                    collisionProto->set_distance(ClampValidNumber(hit.distance));
                    auto* hitPointProto = collisionProto->mutable_hit_point();
                    hitPointProto->set_x(ClampValidNumber(hit.hitPoint.x));
                    hitPointProto->set_y(ClampValidNumber(hit.hitPoint.y));
                    hitPointProto->set_z(ClampValidNumber(hit.hitPoint.z));
                    collisionProto->set_is_closest(i == 0);
                }
            }

            std::ofstream file(filePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "错误：无法导出射线碰撞数据至 " << filePath << std::endl;
                return;
            }
            if (!rayData.SerializeToOstream(&file)) {
                std::cerr << "错误：射线碰撞数据Protobuf序列化失败" << std::endl;
            }
            file.close();
            std::cout << "  Ray collision data exported to: " << filePath << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "错误：射线碰撞数据Protobuf导出失败 - " << e.what() << std::endl;
        }
    }

    void ShowTimeComparison(double aabbInit, double sphereInit, double aabbDetect, double sphereDetect) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nBVH加速方法耗时对比（ms）" << std::endl;
        std::cout << "  方法     初始化耗时  检测耗时   " << std::endl;
        std::cout << " AABB-BVH  " << std::setw(10) << aabbInit << "  " << std::setw(10) << aabbDetect << " " << std::endl;
        std::cout << " 包围球-BVH " << std::setw(9) << sphereInit << "  " << std::setw(10) << sphereDetect << " " << std::endl;
    }

    void ShowHitResult(const std::string& method, const RayIntersectResult& result) {
        std::cout << "\n" << method << "：" << std::endl;
        if (result.hasAnyHit) {
            std::cout << "  - 检测到 " << result.allHits.size() << " 个相交面元" << std::endl;
            const auto& closest = result.GetClosestHit();
            std::cout << "  - 最近碰撞距离：" << closest.distance << " m" << std::endl;
            std::cout << "  - 最近碰撞三角形索引：" << closest.triangleIndex << std::endl;
        }
        else {
            std::cout << "  - 未检测到任何碰撞" << std::endl;
        }
    }
}