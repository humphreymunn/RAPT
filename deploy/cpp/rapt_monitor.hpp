// RAPT runtime monitor — C++ / onnxruntime, no other dependencies.
//
// Loads a checkpoint directory produced by the Python tooling (rapt.onnx +
// deploy_config.csv + obs_stats.csv + calibration.csv) and evaluates the
// hierarchical OOD gates once per control step. Mirrors the on-robot G1
// integration from the paper (~1.6 ms/step at 50 Hz).
//
// Usage:
//   RaptMonitor monitor("checkpoints/robot");
//   monitor.reset();                       // at episode start
//   auto r = monitor.step(obs, action);    // every control tick
//   if (r.is_anomaly) { /* trigger safety response */ }

#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rapt {

struct MonitorResult {
  double risk = 0.0;  // > 1.0 => OOD
  bool is_anomaly = false;
  double local_risk = 0.0;   // per-dimension gate (localized spikes)
  double global_risk = 0.0;  // mean-NLL gate (systemic drift)
  double range_risk = 0.0;   // physical range gate
  int top_dim = 0;
  std::vector<double> per_dim_loss;
};

class RaptMonitor {
 public:
  explicit RaptMonitor(const std::string& checkpoint_dir, bool use_range = true,
                       int threads = 1)
      : use_range_(use_range),
        env_(ORT_LOGGING_LEVEL_WARNING, "rapt"),
        allocator_() {
    auto cfg = loadKeyValueCsv(checkpoint_dir + "/deploy_config.csv");
    obs_dim_ = static_cast<int>(cfg.at("obs_dim"));
    action_dim_ = static_cast<int>(cfg.at("action_dim"));
    embed_dim_ = static_cast<int>(cfg.at("embed_dim"));
    dynamics_ = cfg.at("train_dynamics") != 0.0;

    auto stats = loadRowCsv(checkpoint_dir + "/obs_stats.csv");
    mean_ = stats.at("mean");
    std_ = stats.at("std");

    auto cal = loadRowCsv(checkpoint_dir + "/calibration.csv");
    per_dim_thresh_ = cal.at("per_dim_thresh");
    mean_thresh_ = cal.at("mean_thresh").at(0);
    if (cal.count("risk_scale")) risk_scale_ = cal.at("risk_scale").at(0);
    double buffer = cal.at("range_buffer").at(0);
    const auto& obs_min = cal.at("obs_min");
    const auto& obs_max = cal.at("obs_max");
    range_lo_.resize(obs_dim_);
    range_hi_.resize(obs_dim_);
    range_width_.resize(obs_dim_);
    for (int i = 0; i < obs_dim_; ++i) {
      double span = obs_max[i] - obs_min[i];
      range_lo_[i] = obs_min[i] - buffer * span;
      range_hi_[i] = obs_max[i] + buffer * span;
      range_width_[i] = buffer * span + kEps;
    }

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(
        env_, (checkpoint_dir + "/rapt.onnx").c_str(), opts);
    reset();
  }

  void reset() {
    hidden_.assign(static_cast<size_t>(embed_dim_), 0.0f);
    have_prev_ = false;
  }

  MonitorResult step(const std::vector<float>& obs,
                     const std::vector<float>& action = {}) {
    if (static_cast<int>(obs.size()) != obs_dim_)
      throw std::runtime_error("obs size mismatch");
    std::vector<float> o_norm(obs_dim_);
    for (int i = 0; i < obs_dim_; ++i)
      o_norm[i] = (obs[i] - static_cast<float>(mean_[i])) /
                  static_cast<float>(std_[i]);

    std::vector<float> input;
    std::vector<float> target;
    if (dynamics_) {
      if (static_cast<int>(action.size()) != action_dim_)
        throw std::runtime_error("forward-dynamics checkpoint needs an action");
      if (!have_prev_) {
        prev_obs_ = o_norm;
        prev_action_ = action;
        have_prev_ = true;
        return evaluate(std::vector<double>(obs_dim_, 0.0), obs);
      }
      input = prev_obs_;
      input.insert(input.end(), prev_action_.begin(), prev_action_.end());
      target = o_norm;
      prev_obs_ = o_norm;
      prev_action_ = action;
    } else {
      input = o_norm;
      target = o_norm;
    }

    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> in_shape{1, static_cast<int64_t>(input.size())};
    std::array<int64_t, 3> h_shape{1, 1, static_cast<int64_t>(embed_dim_)};
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                        in_shape.data(), in_shape.size()),
        Ort::Value::CreateTensor<float>(memory, hidden_.data(), hidden_.size(),
                                        h_shape.data(), h_shape.size())};
    const char* in_names[] = {"input", "hidden_in"};
    const char* out_names[] = {"reconstruction", "hidden_out"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names,
                                 inputs.data(), inputs.size(), out_names, 2);

    const float* recon = outputs[0].GetTensorData<float>();
    const float* h_out = outputs[1].GetTensorData<float>();
    std::copy(h_out, h_out + embed_dim_, hidden_.begin());

    std::vector<double> loss(obs_dim_);
    for (int i = 0; i < obs_dim_; ++i) {
      double mu = recon[i];
      double log_var = recon[obs_dim_ + i];
      double diff = target[i] - mu;
      loss[i] = std::exp(-log_var) * diff * diff;
    }
    return evaluate(loss, obs);
  }

  int obsDim() const { return obs_dim_; }
  int actionDim() const { return action_dim_; }
  bool isDynamics() const { return dynamics_; }

 private:
  static constexpr double kEps = 1e-8;

  MonitorResult evaluate(const std::vector<double>& loss,
                         const std::vector<float>& obs) const {
    MonitorResult r;
    r.per_dim_loss = loss;
    double mean_loss = 0.0;
    for (int i = 0; i < obs_dim_; ++i) {
      double ratio = loss[i] / (per_dim_thresh_[i] + kEps);
      if (ratio > r.local_risk) {
        r.local_risk = ratio;
        r.top_dim = i;
      }
      mean_loss += loss[i];
    }
    mean_loss /= obs_dim_;
    r.global_risk = mean_loss / (mean_thresh_ + kEps);
    if (use_range_) {
      for (int i = 0; i < obs_dim_; ++i) {
        double excess = std::max(range_lo_[i] - obs[i], obs[i] - range_hi_[i]);
        r.range_risk = std::max(r.range_risk, 1.0 + excess / range_width_[i]);
      }
    }
    r.risk = std::max({r.local_risk, r.global_risk, r.range_risk}) / risk_scale_;
    r.is_anomaly = r.risk > 1.0;
    return r;
  }

  static std::map<std::string, double> loadKeyValueCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::map<std::string, double> out;
    std::string line;
    while (std::getline(f, line)) {
      auto comma = line.find(',');
      if (comma == std::string::npos) continue;
      out[line.substr(0, comma)] = std::stod(line.substr(comma + 1));
    }
    return out;
  }

  static std::map<std::string, std::vector<double>> loadRowCsv(
      const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::map<std::string, std::vector<double>> out;
    std::string line;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string key, cell;
      std::getline(ss, key, ',');
      std::vector<double> vals;
      while (std::getline(ss, cell, ',')) vals.push_back(std::stod(cell));
      out[key] = std::move(vals);
    }
    return out;
  }

  int obs_dim_ = 0, action_dim_ = 0, embed_dim_ = 0;
  bool dynamics_ = false, use_range_ = true, have_prev_ = false;
  std::vector<double> mean_, std_, per_dim_thresh_, range_lo_, range_hi_,
      range_width_;
  double mean_thresh_ = 0.0;
  double risk_scale_ = 1.0;
  std::vector<float> hidden_, prev_obs_, prev_action_;
  Ort::Env env_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace rapt
