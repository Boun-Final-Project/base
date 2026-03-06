#include "adsm/frontier_finder.h"
#include <cmath>

FrontierFinder::FrontierFinder() {}

FrontierFinder::FrontierFinder(double find_dis_th, double obs_r)
    : find_dis_th_(find_dis_th), obs_r_(obs_r) {
}

double FrontierFinder::distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

std::vector<std::pair<double, double>> FrontierFinder::find(Gridmap* gridmap, double start_x, double start_y, bool only_first) {
    std::lock_guard<std::mutex> lock(gridmap->getMutex());

    int width = gridmap->getSizeInCellsX();
    int height = gridmap->getSizeInCellsY();
    double resolution = gridmap->getResolution();
    int obs_r = (int)(obs_r_ / gridmap->getResolution());

    std::vector<std::pair<double, double>> frontier;
    std::queue<std::pair<int, int>> q;
    std::vector<std::vector<bool>> visited(width, std::vector<bool>(height, false));
    std::vector<std::vector<int>> cost_grid(width, std::vector<int>(height, false));

    int start_x_cell, start_y_cell;
    gridmap->worldToMap(start_x, start_y, start_x_cell, start_y_cell);
    q.push({start_x_cell, start_y_cell});
    visited[start_x_cell][start_y_cell] = true;
    cost_grid[start_x_cell][start_y_cell] = 0;

    int dx[] = {0, 1, 0, -1};
    int dy[] = {1, 0, -1, 0};

    bool find_one = false;
    while (!q.empty()) {
        double x = q.front().first;
        double y = q.front().second;
        q.pop();

        int cost = gridmap->getCost(x, y);
        if (cost == Gridmap::NO_INFORMATION || cost == Gridmap::LETHAL_OBSTACLE) {
            continue;
        }

        int x_lim_s = std::max(0, static_cast<int>(x) - obs_r);
        int x_lim_e = std::min(static_cast<int>(width), static_cast<int>(x) + obs_r);
        int y_lim_s = std::max(0, static_cast<int>(y) - obs_r);
        int y_lim_e = std::min(static_cast<int>(height), static_cast<int>(y) + obs_r);
        bool obs_check = false;
        for (int i = x_lim_s; i <= x_lim_e; ++i) {
            for (int j = y_lim_s; j <= y_lim_e; ++j) {
                if (gridmap->getCost(i, j) == Gridmap::LETHAL_OBSTACLE) {
                    obs_check = true;
                    break;
                }
            }
            if (obs_check) {
                break;
            }
        }
        if (obs_check) {
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }
            if (!visited[nx][ny]) {
                visited[nx][ny] = true;
                cost_grid[nx][ny] = cost_grid[(int)x][(int)y] + 1;
                if (resolution * cost_grid[nx][ny] > find_dis_th_) {
                    continue;
                }
                int cost = gridmap->getCost(nx, ny);
                if (cost == Gridmap::NO_INFORMATION) {
                    double map_x, map_y;
                    gridmap->mapToWorld(nx, ny, map_x, map_y);
                    frontier.push_back({map_x, map_y});
                    find_one = true;
                    if (only_first) {
                        break;
                    }
                } else if (cost < Gridmap::LETHAL_OBSTACLE) {
                    q.push({nx, ny});
                }
            }
        }
        if (find_one && only_first) {
            break;
        }
    }

    return frontier;
}
