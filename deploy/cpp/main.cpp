// Replay a recorded observation log (observations.csv, optional actions.csv)
// through the RAPT monitor and report detections + per-step latency.
//
//   ./rapt_replay <checkpoint_dir> <log_dir>
//
// CSV format: header row, first column timestamp (dropped), one row per
// control step — the format written by the deployment logger and by
// scripts/train.py's calibration export.

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rapt_monitor.hpp"

static std::vector<std::vector<float>> readCsv(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  std::vector<std::vector<float>> rows;
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (first) {  // skip header
      first = false;
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> row;
    bool ts = true;
    while (std::getline(ss, cell, ',')) {
      if (ts) {  // drop timestamp column
        ts = false;
        continue;
      }
      row.push_back(std::stof(cell));
    }
    if (!row.empty()) rows.push_back(std::move(row));
  }
  return rows;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <checkpoint_dir> <log_dir>\n";
    return 1;
  }
  const std::string ckpt = argv[1];
  const std::string log_dir = argv[2];

  rapt::RaptMonitor monitor(ckpt);
  auto obs = readCsv(log_dir + "/observations.csv");
  auto actions = readCsv(log_dir + "/actions.csv");
  if (obs.empty()) {
    std::cerr << "no observations found in " << log_dir << "\n";
    return 1;
  }
  if (monitor.isDynamics() && actions.size() != obs.size()) {
    std::cerr << "forward-dynamics checkpoint needs aligned actions.csv\n";
    return 1;
  }

  double total_ms = 0.0;
  int first_detection = -1;
  for (size_t t = 0; t < obs.size(); ++t) {
    auto t0 = std::chrono::steady_clock::now();
    auto r = monitor.step(obs[t], monitor.isDynamics() ? actions[t]
                                                       : std::vector<float>{});
    total_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    if (r.is_anomaly && first_detection < 0) {
      first_detection = static_cast<int>(t);
      std::cout << "OOD detected at step " << t << " (risk " << r.risk
                << ", top dim " << r.top_dim << ", gates L/G/R "
                << r.local_risk << "/" << r.global_risk << "/" << r.range_risk
                << ")\n";
    }
  }
  std::cout << "steps: " << obs.size()
            << " | mean latency: " << total_ms / obs.size() << " ms\n";
  if (first_detection < 0) std::cout << "no anomaly detected\n";
  return 0;
}
