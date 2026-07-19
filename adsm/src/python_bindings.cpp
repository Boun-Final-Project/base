#include "adsm/engine.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <random>

namespace py = pybind11;
using adsm_core::Config;
using adsm_core::Decision;
using adsm_core::Engine;
using adsm_core::Grid;
using adsm_core::Input;

PYBIND11_MODULE(adsm_core_py, m) {
  py::class_<Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("k1", &Config::k1)
      .def_readwrite("random_sample_r", &Config::random_sample_r)
      .def_readwrite("goal_cluster_num", &Config::goal_cluster_num)
      .def_readwrite("obs_r", &Config::obs_r)
      .def_readwrite("goal_reach_th", &Config::goal_reach_th)
      .def_readwrite("resample_time_th", &Config::resample_time_th)
      .def_readwrite("gas_high_th", &Config::gas_high_th)
      .def_readwrite("gas_low_th", &Config::gas_low_th)
      .def_readwrite("sensor_window_length", &Config::sensor_window_length)
      .def_readwrite("frontier_search_th", &Config::frontier_search_th)
      .def_readwrite("rrt_max_iter", &Config::rrt_max_iter)
      .def_readwrite("rrt_max_r", &Config::rrt_max_r)
      .def_readwrite("rrt_min_r", &Config::rrt_min_r)
      .def_readwrite("rrt_step_size", &Config::rrt_step_size)
      .def_readwrite("stuck_duration_th", &Config::stuck_duration_th);

  py::class_<Decision>(m, "Decision")
      .def_readonly("x", &Decision::x).def_readonly("y", &Decision::y)
      .def_readonly("yaw", &Decision::yaw).def_readonly("goal_type", &Decision::goal_type)
      .def_readonly("j", &Decision::j).def_readonly("j_p", &Decision::j_p)
      .def_readonly("j_i", &Decision::j_i).def_readonly("iteration", &Decision::iteration)
      .def_readonly("gas_hit", &Decision::gas_hit).def_readonly("resampled", &Decision::resampled)
      .def_readonly("epi_size", &Decision::epi_size).def_readonly("epr_size", &Decision::epr_size);

  py::class_<Engine>(m, "Engine")
      .def(py::init<const Config &>(), py::arg("config") = Config())
      .def("reset", [](Engine &self, py::object seed) {
        uint32_t actual = seed.is_none() ? std::random_device{}() : seed.cast<uint32_t>();
        self.reset(actual); return actual;
      }, py::arg("seed") = py::none())
      .def("step", [](Engine &self, double x, double y, double yaw, double gas,
                       double wind_speed, double wind_direction, double sim_time,
                       py::array_t<int8_t, py::array::c_style | py::array::forcecast> grid,
                       double resolution, double origin_x, double origin_y) {
        auto b = grid.request();
        if (b.ndim != 2) throw py::value_error("occupancy grid must be HxW");
        Grid g; g.height = static_cast<int>(b.shape[0]); g.width = static_cast<int>(b.shape[1]);
        g.resolution = resolution; g.origin_x = origin_x; g.origin_y = origin_y;
        const auto *src = static_cast<const int8_t *>(b.ptr);
        g.data.resize(static_cast<std::size_t>(g.width * g.height));
        for (std::size_t i = 0; i < g.data.size(); ++i)
          g.data[i] = src[i] < 0 ? -1 : (src[i] == 0 ? 0 : 100);
        return self.step(Input{x, y, yaw, gas, wind_speed, wind_direction, sim_time, std::move(g)});
      });
}
