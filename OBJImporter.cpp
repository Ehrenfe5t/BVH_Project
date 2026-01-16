#define _CRT_SECURE_NO_WARNINGS
#include "Core.h"
#include <sstream>
#include <unordered_map>

namespace ModelImporter {
    void TriangulatePolygon(const std::vector<int>& indices, std::vector<std::tuple<int, int, int>>& triangles) {
        if (indices.size() < 3) return;
        for (size_t i = 1; i < indices.size() - 1; ++i) {
            triangles.emplace_back(indices[0], indices[i], indices[i + 1]);
        }
    }

    bool LoadOBJ(const std::string& filePath, Scenario3D& scene, ProgressBar* progressBar) {
        scene.points.clear();
        scene.triangles.clear();
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "错误：无法打开OBJ文件 " << filePath << std::endl;
            return false;
        }

        // 修复：替换 starts_with（C++20特性）为兼容写法，支持C++17及以下
        int estimatedTotalSteps = 0;
        std::string line;
        while (std::getline(file, line)) {
            // 兼容写法：判断字符串是否以 "v " 开头（避免越界，先判断长度）
            bool isVertex = (line.size() >= 2) && (line.substr(0, 2) == "v ");
            // 兼容写法：判断字符串是否以 "f " 开头
            bool isFace = (line.size() >= 2) && (line.substr(0, 2) == "f ");

            if (isVertex) {
                estimatedTotalSteps += 1;
            }
            else if (isFace) {
                estimatedTotalSteps += 3;
            }
        }
        file.clear();
        file.seekg(0, std::ios::beg);

        int processedSteps = 0;
        if (progressBar) {
            progressBar->Update(0);
        }

        std::vector<Point3D> normals;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "v") {
                double x, y, z;
                iss >> x >> y >> z;
                scene.points.emplace_back(x, z, y);
                processedSteps += 1;
            }
            else if (token == "vn") {
                double nx, ny, nz;
                iss >> nx >> ny >> nz;
                normals.emplace_back(nx, nz, ny);
            }
            else if (token == "f") {
                std::vector<int> vertexIndices;
                std::string faceToken;
                while (iss >> faceToken) {
                    std::vector<std::string> parts;
                    std::stringstream ss(faceToken);
                    std::string part;
                    while (std::getline(ss, part, '/')) parts.push_back(part);
                    if (!parts[0].empty()) {
                        int vIdx = std::stoi(parts[0]) - 1;
                        vertexIndices.push_back(vIdx);
                    }
                }
                std::vector<std::tuple<int, int, int>> triIndices;
                TriangulatePolygon(vertexIndices, triIndices);
                for (const auto& tri : triIndices) {
                    int p1 = std::get<0>(tri);
                    int p2 = std::get<1>(tri);
                    int p3 = std::get<2>(tri);
                    Point3D normal;
                    if (!normals.empty() && p1 < normals.size()) {
                        normal = normals[p1];
                    }
                    else {
                        const Point3D& p1p = scene.points[p1];
                        const Point3D& p2p = scene.points[p2];
                        const Point3D& p3p = scene.points[p3];
                        Point3D e1 = p2p - p1p;
                        Point3D e2 = p3p - p1p;
                        normal = GeometryUtils::Cross(e1, e2).Normalize();
                    }
                    scene.triangles.emplace_back(p1, p2, p3, normal);
                }
                processedSteps += 3;
            }

            if (progressBar && processedSteps % 100 == 0) {
                progressBar->Update(100);
            }
        }

        if (progressBar) {
            int remainingSteps = estimatedTotalSteps - processedSteps;
            if (remainingSteps > 0) {
                progressBar->Update(remainingSteps);
            }
            progressBar->Finish();
        }

        file.close();
        std::cout << "成功加载OBJ文件：" << filePath << std::endl;
        std::cout << "  顶点数：" << scene.points.size() << std::endl;
        std::cout << "  三角形数：" << scene.triangles.size() << std::endl;
        return true;
    }
}