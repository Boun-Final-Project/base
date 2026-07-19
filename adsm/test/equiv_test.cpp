// ADSM port faithfulness: input-output equivalence test.
//
// Both implementations' decision math is extracted VERBATIM from source
// (only the member variables become explicit parameters, since the port
// binds real_x_/real_y_/real_yaw_ where the original binds x_/y_/yaw_).
// Identical inputs are fed to both; outputs are compared bit-exact.
//
//   orig::  /tmp/.../adsm_orig/src/adsm.cpp      (paper, ROS1)
//   port::  ~/ros2_ws/src/base/adsm/src/adsm.cpp (our ROS2 port)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

// ----------------------------------------------------------------------------
// 1) probability()  — the Gaussian indoor plume / source-term estimator (Eq.3)
// ----------------------------------------------------------------------------
namespace orig {
// verbatim from adsm_orig/src/adsm.cpp:199-217 (x_, y_, yaw_ -> params)
double probability(double x, double y, double x_, double y_, double yaw_,
                   double wind_speed_, double wind_direction_) {
    double Q = 4.0;
    double D = 1.0;
    double tau = 250;
    double V = wind_speed_;
    double phi = yaw_ - wind_direction_;
    double lam = std::sqrt(D * tau / (1 + V * V * tau / (4 * D)));
    double dis = std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
    double dx = x_ - x;
    double dy = y_ - y;

    double pa = Q / (4 * M_PI * D * (std::abs(dis + 0.0001)));
    double pb = std::exp(-dis / lam);
    double pc = std::exp(-dx * V * std::cos(phi) / (2 * D));
    double pd = std::exp(-dy * V * std::sin(phi) / (2 * D));
    double p = pa * pb * pc * pd;

    return p;
}
} // namespace orig

namespace port {
// verbatim from src/base/adsm/src/adsm.cpp:361-379 (real_x_, real_y_, real_yaw_ -> params)
double probability(double x, double y, double real_x_, double real_y_, double real_yaw_,
                   double wind_speed_, double wind_direction_) {
    double Q = 4.0;
    double D = 1.0;
    double tau = 250;
    double V = wind_speed_;
    double phi = real_yaw_ - wind_direction_;
    double lam = std::sqrt(D * tau / (1 + V * V * tau / (4 * D)));
    double dis = std::sqrt((x - real_x_) * (x - real_x_) + (y - real_y_) * (y - real_y_));
    double dx = real_x_ - x;
    double dy = real_y_ - y;

    double pa = Q / (4 * M_PI * D * (std::abs(dis + 0.0001)));
    double pb = std::exp(-dis / lam);
    double pc = std::exp(-dx * V * std::cos(phi) / (2 * D));
    double pd = std::exp(-dy * V * std::sin(phi) / (2 * D));
    double p = pa * pb * pc * pd;

    return p;
}
} // namespace port

// ----------------------------------------------------------------------------
// 2) evaluate() — fitness j = j_p + k1*f*j_i, then argmax goal selection
// ----------------------------------------------------------------------------
struct Goal { double iteration, x, y, probability, frontier_size, j, j_p, j_i; int type; };

namespace orig {
// verbatim from adsm_orig/src/adsm.cpp:416-452
int evaluate(std::vector<Goal>& goals_, int iter_, double k1_) {
    double sum_probability = 0.0, sum_frontier_size = 0.0;
    for (auto& g : goals_) { sum_probability += g.probability; sum_frontier_size += g.frontier_size; }
    for (auto& temp_goal : goals_) {
        temp_goal.j_p = sum_probability > 0 ? temp_goal.probability / sum_probability : 0.0;
        temp_goal.j_i = sum_frontier_size > 0 ? temp_goal.frontier_size / sum_frontier_size : 0.0;
        double f = temp_goal.iteration / static_cast<double>(iter_);
        temp_goal.j = temp_goal.j_p + k1_ * f * temp_goal.j_i;
    }
    int best = 0;
    for (size_t i = 0; i < goals_.size(); ++i) if (goals_[i].j > goals_[best].j) best = (int)i;
    return best;
}
} // namespace orig

namespace port {
// verbatim from src/base/adsm/src/adsm.cpp:598-645 (same math; the port adds a
// `sum_p==0 && sum_f==0 -> random` branch, exercised separately below)
int evaluate(std::vector<Goal>& goals_, int iter_, double k1_, bool* took_random_branch) {
    double sum_probability = 0.0, sum_frontier_size = 0.0;
    for (auto& g : goals_) { sum_probability += g.probability; sum_frontier_size += g.frontier_size; }
    if (sum_probability == 0.0 && sum_frontier_size == 0.0) { *took_random_branch = true; return -1; }
    *took_random_branch = false;
    for (auto& temp_goal : goals_) {
        temp_goal.j_p = sum_probability > 0 ? temp_goal.probability / sum_probability : 0.0;
        temp_goal.j_i = sum_frontier_size > 0 ? temp_goal.frontier_size / sum_frontier_size : 0.0;
        double f = temp_goal.iteration / static_cast<double>(iter_);
        temp_goal.j = temp_goal.j_p + k1_ * f * temp_goal.j_i;
    }
    int best = 0;
    for (size_t i = 0; i < goals_.size(); ++i) if (goals_[i].j > goals_[best].j) best = (int)i;
    return best;
}
} // namespace port

// ----------------------------------------------------------------------------
// 3) gas binarization (gas_hit) — KNOWN intentional difference: MOX -> PID
// ----------------------------------------------------------------------------
namespace orig {
// verbatim from adsm_orig/src/adsm.cpp:164-191
bool gas_hit(double raw, double gas_max_, double gas_low_th_, double gas_high_th_,
             const std::vector<double>& window, double* gas_out) {
    double gas_ = gas_max_ - raw;          // MOX: resistance inverted
    *gas_out = gas_;
    if (gas_ < gas_low_th_) return false;
    if (gas_ > gas_high_th_) return true;
    bool in_rec = true;
    for (size_t i = 1; i < window.size(); ++i)
        if (window[i] - window[i-1] >= 0.0 || window[i-1] - gas_high_th_ >= 0.0) in_rec = false;
    return in_rec ? false : true;
}
} // namespace orig

namespace port {
// verbatim from src/base/adsm/src/adsm.cpp:327-353
bool gas_hit(double raw, double gas_max_, double gas_low_th_, double gas_high_th_,
             const std::vector<double>& window, double* gas_out) {
    double gas_ = raw;                     // PID: direct ppm (no inversion)
    *gas_out = gas_;
    if (gas_ < gas_low_th_) return false;
    if (gas_ > gas_high_th_) return true;
    bool in_rec = true;
    for (size_t i = 1; i < window.size(); ++i)
        if (window[i] - window[i-1] >= 0.0 || window[i-1] - gas_high_th_ >= 0.0) in_rec = false;
    return in_rec ? false : true;
}
} // namespace port

// ----------------------------------------------------------------------------
// 4) estimate() angular clustering: N classes by bearing, farthest-per-class
// ----------------------------------------------------------------------------
struct Node { double x, y; };
namespace orig {
// verbatim from adsm_orig/src/adsm.cpp:298-330 (x_, y_ -> params)
std::vector<int> farthest_per_class(const std::vector<Node>& n, double x_, double y_, int K) {
    std::vector<int> categories(n.size(), -1);
    for (size_t i = 0; i < n.size(); ++i) {
        if (i == 0) continue;
        double angle = atan2(n[i].y - y_, n[i].x - x_);
        if (angle < 0) angle += 2 * M_PI;
        categories[i] = static_cast<int>((angle / (2 * M_PI)) * K);
    }
    std::vector<double> farthest_length(K, -1.0);
    std::vector<int> fartest_index(K, -1);
    for (size_t i = 0; i < n.size(); ++i) {
        if (categories[i] < 0) continue;
        double dist = std::sqrt((n[i].x-x_)*(n[i].x-x_) + (n[i].y-y_)*(n[i].y-y_));
        if (dist > farthest_length[categories[i]]) { farthest_length[categories[i]] = dist; fartest_index[categories[i]] = (int)i; }
    }
    return fartest_index;
}
} // namespace orig
namespace port {
// verbatim from src/base/adsm/src/adsm.cpp:479-512 (real_x_, real_y_ -> params)
std::vector<int> farthest_per_class(const std::vector<Node>& n, double real_x_, double real_y_, int K) {
    std::vector<int> categories(n.size(), -1);
    for (size_t i = 0; i < n.size(); ++i) {
        if (i == 0) continue;
        double angle = atan2(n[i].y - real_y_, n[i].x - real_x_);
        if (angle < 0) angle += 2 * M_PI;
        categories[i] = static_cast<int>((angle / (2 * M_PI)) * K);
    }
    std::vector<double> farthest_length(K, -1.0);
    std::vector<int> fartest_index(K, -1);
    for (size_t i = 0; i < n.size(); ++i) {
        if (categories[i] < 0) continue;
        double dist = std::sqrt((n[i].x-real_x_)*(n[i].x-real_x_) + (n[i].y-real_y_)*(n[i].y-real_y_));
        if (dist > farthest_length[categories[i]]) { farthest_length[categories[i]] = dist; fartest_index[categories[i]] = (int)i; }
    }
    return fartest_index;
}
} // namespace port

// ----------------------------------------------------------------------------
int main() {
    std::mt19937_64 rng(12345);
    auto U = [&](double a, double b){ return a + (b-a) * (double)(rng() % 1000000) / 1000000.0; };

    // --- Test 1: probability() over a large random sweep -------------------
    long n1 = 2000000, mismatch1 = 0; double maxdiff1 = 0;
    for (long i = 0; i < n1; ++i) {
        double cx=U(-10,10), cy=U(-10,10), rx=U(-10,10), ry=U(-10,10),
               yaw=U(-M_PI,M_PI), ws=U(0,3), wd=U(-M_PI,M_PI);
        double a = orig::probability(cx,cy,rx,ry,yaw,ws,wd);
        double b = port::probability(cx,cy,rx,ry,yaw,ws,wd);
        if (!(a==b)) { if (!(std::isnan(a)&&std::isnan(b))) { mismatch1++; maxdiff1 = std::max(maxdiff1, std::abs(a-b)); } }
    }
    printf("1) probability()       : %ld/%ld inputs, bit-exact mismatches = %ld  (max|diff| = %g)\n",
           n1, n1, mismatch1, maxdiff1);

    // --- Test 2: evaluate() fitness + argmax --------------------------------
    long n2 = 200000, mismatch2 = 0, randbranch = 0;
    for (long i = 0; i < n2; ++i) {
        int m = 1 + (int)(rng() % 40);
        int iter = 1 + (int)(rng() % 400);
        double k1 = U(0,1);
        std::vector<Goal> ga(m), gb;
        for (auto& g : ga) {
            g.iteration = (double)(1 + (int)(rng()%iter));
            g.probability = (rng()%5==0) ? 0.0 : U(0,1e-3);
            g.frontier_size = (rng()%5==0) ? 0.0 : (double)(rng()%50);
            g.x=U(-5,5); g.y=U(-5,5); g.type=0; g.j=g.j_p=g.j_i=0;
        }
        gb = ga;
        int ia = orig::evaluate(ga, iter, k1);
        bool rb=false; int ib = port::evaluate(gb, iter, k1, &rb);
        if (rb) { randbranch++; continue; }          // port-only branch, compared separately
        if (ia != ib) mismatch2++;
        else for (int q=0;q<m;q++) if (!(ga[q].j==gb[q].j)) { mismatch2++; break; }
    }
    printf("2) evaluate() j+argmax : %ld cases, selected-goal/j mismatches = %ld  (port-only random branch hit %ld x)\n",
           n2-randbranch, mismatch2, randbranch);

    // --- Test 3: gas binarization (expected to DIFFER: MOX vs PID) ---------
    long n3 = 200000, same3 = 0;
    for (long i = 0; i < n3; ++i) {
        double raw = U(0, 5);
        std::vector<double> w; for (int k=0;k<6;k++) w.push_back(U(0,1));
        double ga_, gb_;
        bool a = orig::gas_hit(raw, 63000.0, 500.0, 2000.0, w, &ga_);   // paper's MOX params
        bool b = port::gas_hit(raw, 10.0, 0.1, 0.3, w, &gb_);           // our PID params
        if (a==b) same3++;
    }
    printf("3) gas binarization    : agreement %ld/%ld  <-- EXPECTED to differ (MOX->PID, intentional)\n", same3, n3);

    // --- Test 4: angular clustering / farthest-per-class -------------------
    long n4 = 100000, mismatch4 = 0;
    for (long i = 0; i < n4; ++i) {
        int m = 2 + (int)(rng() % 60); int K = 4 + (int)(rng()%32);
        double rx=U(-5,5), ry=U(-5,5);
        std::vector<Node> nd(m);
        for (auto& p : nd) { p.x=U(-8,8); p.y=U(-8,8); }
        auto a = orig::farthest_per_class(nd, rx, ry, K);
        auto b = port::farthest_per_class(nd, rx, ry, K);
        if (a != b) mismatch4++;
    }
    printf("4) angular clustering  : %ld cases, farthest-per-class mismatches = %ld\n", n4, mismatch4);

    printf("\nVERDICT: decision core %s\n",
        (mismatch1==0 && mismatch2==0 && mismatch4==0)
        ? "BIT-EXACT EQUIVALENT (only the documented MOX->PID sensor conversion differs)"
        : "*** DIVERGENCE FOUND ***");
    return 0;
}
