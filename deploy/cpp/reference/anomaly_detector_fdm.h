// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
#pragma once

#include <onnxruntime_cxx_api.h>
#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <queue>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <H5Cpp.h>

namespace isaaclab
{

class AnomalyDetectorFDM
{
public:
    struct ConfigFDM
    {
        std::string ae_model_path;        // Path to exported MAE ONNX model
        std::string obs_stats_path;        // Path to observation stats
        std::string calibration_loss_file = ""; // Path to calibration loss file
        bool enable_inference = true;      // Enable model inference/loss computation
        int check_frequency = 1;           // Check every N steps
        int range_timestep_offset = 0;     // Kept for backward compatibility (unused by range detector)
        bool enable_logging = true;        // Enable detailed logging
        std::string log_dir = "logs/anomaly_detection";
        int obs_dim;                     // Observation dimension [96 velocity, 154 mimic]
        int action_dim = 29;
        int hidden_dim = 256;
    };

    struct AnomalyResult
    {
        bool is_anomaly = false;
        bool is_dim_loss_anomaly = false;
        bool is_mean_loss_anomaly = false;
        bool is_range_anomaly = false;

        float current_mean_loss = 0.0f;
        float limit_mean_loss = 0.0f;

        std::vector<float> threshold_delta_loss;
        std::vector<float> losses;
        std::vector<float> values;
        std::vector<int> top_anomaly_indices;
        std::vector<std::string> top_anomaly_names;
        std::vector<float> top_anomaly_values;
        float reconstruction_loss_mean = 0.0f;
        float reconstruction_loss_max = 0.0f;
        int timestep = 0;
        int range_row_index = 0;
        std::vector<float> reconstruction_mu;      
        std::vector<float> reconstruction_logvar; 
        std::vector<float> top_anomaly_loss_limits;
        std::vector<float> top_anomaly_range_mins;
        std::vector<float> top_anomaly_range_maxs;
        std::vector<float> top_anomaly_mse_over_var_medians;
    };

private:
    std::vector<float> gru_hidden_state_;

    ConfigFDM config_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Session> ae_session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    
    // Normalization parameters
    Eigen::VectorXf obs_mean_;
    Eigen::VectorXf obs_std_;
    Eigen::MatrixXf obs_min_ts_;
    Eigen::MatrixXf obs_max_ts_;
    Eigen::VectorXf obs_min_global_;
    Eigen::VectorXf obs_max_global_;

    std::vector<float> calib_thresholds_;  // Stores max NLL + 5*Std
    std::vector<float> calib_max_err_;     // Per-dim Max NLL
    std::vector<float> calib_range_err_;   // Per-dim Range (Max - Min) for Range Detector
    std::vector<float> calib_median_mse_over_var_; // Per-dim median from calibration mse_over_var.csv
    
    float calib_max_mean_err_ = 10000.0f;   // Global Mean Max
    float calib_range_mean_err_ = 1.0f;    // Global Mean Range
    
    // Observation names mapping
    std::vector<std::string> obs_names_;
    
    // Logging
    std::ofstream log_file_;
    std::ofstream transition_log_file_;

public:
    AnomalyDetectorFDM(const ConfigFDM& config)
        : config_(config),
          env_(ORT_LOGGING_LEVEL_WARNING, "AnomalyDetectorFDM"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        initialize();
    }

    ~AnomalyDetectorFDM()
    {
        if (log_file_.is_open())
        {
            log_file_.close();
        }
        if (transition_log_file_.is_open())
        {
            transition_log_file_.close();
        }
    }

    void resetState() {
        gru_hidden_state_.assign(config_.hidden_dim, 0.0f);
    }

    void initialize()
    {
        if (config_.obs_dim == 140) {
            config_.action_dim = 43;
        } else {
            config_.action_dim = 29;
        }
        calib_thresholds_.assign(config_.obs_dim, 10000.0f);
        calib_max_err_.assign(config_.obs_dim, 10000.0f);
        calib_range_err_.assign(config_.obs_dim, 1.0f);
        calib_median_mse_over_var_.assign(config_.obs_dim, 1.0f);

        // Initialize observation names
        initializeObservationNames();
        
        if (config_.enable_inference)
        {
            // Load ONNX model
            loadAEModel();
            
            // Load normalization statistics
            loadObservationStats();

            if (!config_.calibration_loss_file.empty()) {
                loadCalibrationLosses();
            } else {
                std::cerr << "[AnomalyDetectorFDM] WARNING: No calibration file! Logic will fail." << std::endl;
            }
        }
        else
        {
            std::cout << "[AnomalyDetectorFDM] Logging-only mode (inference disabled)" << std::endl;
        }
        
        // Setup logging
        if (config_.enable_logging)
        {
            setupLogging();
        }
        resetState();
        std::cout << "  - Hidden Dim: " << config_.hidden_dim << std::endl; // Debug print
        std::cout << "[AnomalyDetectorFDM] Initialization complete" << std::endl;
        std::cout << "  - AE Model: " << config_.ae_model_path << std::endl;
        std::cout << "  - Observation dimension: " << config_.obs_dim << std::endl;
        std::cout << "  - Check frequency: " << config_.check_frequency << std::endl;
    }

    void loadCalibrationLosses()
    {
        auto buildSiblingFilePath = [](const std::string& path, const std::string& sibling_name) {
            size_t slash_pos = path.find_last_of("/");
            if (slash_pos == std::string::npos) {
                return sibling_name;
            }
            return path.substr(0, slash_pos + 1) + sibling_name;
        };

        std::string calibration_file = config_.calibration_loss_file;
        std::string preferred_mse_file = buildSiblingFilePath(config_.calibration_loss_file, "mse_over_var.csv");

        std::ifstream preferred_file(preferred_mse_file);
        if (preferred_file.is_open()) {
            calibration_file = preferred_mse_file;
            preferred_file.close();
        }

        std::cout << "[AnomalyDetectorFDM] Loading Calibration: " << calibration_file << std::endl;
        std::ifstream file(calibration_file);
        if (!file.is_open()) {
            std::cerr << "[AnomalyDetectorFDM] ERROR: Could not open calibration file. Using safe defaults." << std::endl;
            return;
        }

        std::string line;
        std::getline(file, line); // Skip Header

        std::vector<std::vector<float>> history(config_.obs_dim);
        int row_count = 0;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;
            std::getline(ss, cell, ','); // Skip timestamp

            for (int i = 0; i < config_.obs_dim; i++)
            {
                if (std::getline(ss, cell, ',')) {
                    try {
                        float val = std::stof(cell);
                        history[i].push_back(val);
                    } catch (...) {}
                }
            }
            row_count++;
        }
        if (row_count < 10) {
            std::cerr << "[AnomalyDetectorFDM] WARNING: Calibration file too short/empty. Defaults retained." << std::endl;
            return;
        }

        calib_median_mse_over_var_.assign(config_.obs_dim, 1.0f);
        for (int i = 0; i < config_.obs_dim; ++i) {
            if (history[i].empty()) {
                calib_median_mse_over_var_[i] = 1.0f;
                continue;
            }

            std::vector<float> sorted_vals = history[i];
            std::sort(sorted_vals.begin(), sorted_vals.end());
            const size_t n = sorted_vals.size();
            float median = 1.0f;
            if (n % 2 == 0) {
                median = 0.5f * (sorted_vals[n / 2 - 1] + sorted_vals[n / 2]);
            } else {
                median = sorted_vals[n / 2];
            }
            if (median < 1e-6f || !std::isfinite(median)) {
                median = 1.0f;
            }
            calib_median_mse_over_var_[i] = median;
        }

        std::vector<float> mean_history;
        int history_rows = 0;
        for (const auto& dim_hist : history) {
            history_rows = std::max(history_rows, static_cast<int>(dim_hist.size()));
        }
        for (int row = 0; row < history_rows; ++row) {
            float row_sum = 0.0f;
            int count = 0;
            for (int i = 0; i < config_.obs_dim; ++i) {
                if (row >= static_cast<int>(history[i].size())) {
                    continue;
                }
                float median = calib_median_mse_over_var_[i];
                if (median < 1e-6f) median = 1.0f;
                row_sum += history[i][row];
                count++;
            }
            if (count > 0) {
                mean_history.push_back(row_sum / static_cast<float>(count));
            }
        }

        calib_thresholds_.assign(config_.obs_dim, 10000.0f);
        calib_max_err_.assign(config_.obs_dim, 10000.0f);
        calib_range_err_.assign(config_.obs_dim, 0.0f);

        for(int i=0; i<config_.obs_dim; ++i) {
            if(history[i].empty()) {
                calib_max_err_[i] = 100.0f; // High default
                continue;
            }

            std::vector<float> scaled_history;
            scaled_history.reserve(history[i].size());
            float median = calib_median_mse_over_var_[i];
            if (median < 1e-6f) median = 1.0f;
            for (float val : history[i]) {
                scaled_history.push_back(val);
            }

            double sum = std::accumulate(scaled_history.begin(), scaled_history.end(), 0.0);
            double mean = sum / scaled_history.size();

            // calculate std dev of error
            double sq_sum = 0.0;
            for(float val : scaled_history) {
                sq_sum += (val - mean) * (val - mean);
            }
            double std_dev = std::sqrt(sq_sum / scaled_history.size());

            // find max
            auto [min_it, max_it] = std::minmax_element(scaled_history.begin(), scaled_history.end());
            float max_val = *max_it;
            float range = *max_it - *min_it;

            std::vector<float> sorted_scaled = scaled_history;
            std::sort(sorted_scaled.begin(), sorted_scaled.end());
            const size_t n_scaled = sorted_scaled.size();
            float median_loss = 0.0f;
            if (n_scaled % 2 == 0) {
                median_loss = 0.5f * (sorted_scaled[n_scaled / 2 - 1] + sorted_scaled[n_scaled / 2]);
            } else {
                median_loss = sorted_scaled[n_scaled / 2];
            }

            calib_max_err_[i] = max_val;
            calib_range_err_[i] = range;

            // set threshold: max + 0.25 * (max - median)
            float robust_limit = max_val + ((max_val - median_loss) * 2.0f);
            calib_thresholds_[i] = robust_limit;

            // Debug first few dims
            if (i < 3) {
                std::cout << "  - Dim " << i << ": Mean=" << mean << ", Std=" << std_dev 
                          << ", Max=" << max_val << " -> Threshold=" << calib_thresholds_[i] << std::endl;
                std::cout << "    median(mse_over_var)=" << calib_median_mse_over_var_[i] << std::endl;
            }
        }

        // calculate mean stats
        if(!mean_history.empty()) {
            double sum = std::accumulate(mean_history.begin(), mean_history.end(), 0.0);
            double mean = sum / mean_history.size();
            
            double sq_sum = 0.0;
            for(float val : mean_history) sq_sum += (val - mean) * (val - mean);
            double std_dev = std::sqrt(sq_sum / mean_history.size());

            // Limit for Mean Loss (max + 1 sigma)
            float max_mean_val = *std::max_element(mean_history.begin(), mean_history.end());
            calib_max_mean_err_ = max_mean_val + (3.0f * static_cast<float>(std_dev));

            std::cout << "  - Global Mean Max: " << max_mean_val << " -> Limit: " << calib_max_mean_err_ << std::endl;
        }

        std::cout << "  - Calibrated Mean Max: " << calib_max_mean_err_ << std::endl;
    }

    
    const ConfigFDM& getConfig() const { return config_; }

    const std::vector<std::string>& getObsNames() const { return obs_names_; }

    int getRangeRowIndex(int timestep) const
    {
        (void)timestep;
        return 0;
    }

    void getRangeBounds(int idx, int timestep, float& out_min, float& out_max, float buffer_scale = 1.0f) const
    {
        (void)timestep;
        const Eigen::VectorXf& range_min = obs_min_global_;
        const Eigen::VectorXf& range_max = obs_max_global_;
        if (idx < 0 || idx >= config_.obs_dim) {
            out_min = 0.0f;
            out_max = 0.0f;
            return;
        }
        float range = range_max[idx] - range_min[idx];
        if (range < 1e-6f) range = 1.0f;
        float buffer = range * buffer_scale;
        out_min = range_min[idx] - buffer;
        out_max = range_max[idx] + buffer;
    }

    AnomalyResult checkTransition(const Eigen::VectorXf& prev_obs, 
                                  const std::vector<float>& prev_action, 
                                  const Eigen::VectorXf& curr_obs, 
                                  int timestep)
    {
        AnomalyResult result;
        result.timestep = timestep;
        result.range_row_index = 0;
        
        if (timestep % config_.check_frequency != 0) return result;
        if (!config_.enable_inference) return result;
        
        // A. Normalize Obs
        Eigen::VectorXf prev_obs_norm = (prev_obs - obs_mean_).cwiseQuotient(obs_std_);
        
        if (prev_action.size() != config_.action_dim) {
            std::cerr << "[AnomalyDetector] Action Dim Mismatch!" << std::endl;
            return result;
        }
        
        std::vector<float> input_vec;
        input_vec.reserve(config_.obs_dim + config_.action_dim);

        // Copy Obs
        for(int i=0; i<prev_obs_norm.size(); ++i) input_vec.push_back(prev_obs_norm[i]);
        
        
        // Copy Actions (Not normalized, model takes unnormalized, using Obs indices 67 to 95) -> only for velocity task
        // check velocity task: obs_dim=96 (take last 29 dims as actions), for throwing task: obs_dim=140 (take last 43 dims as actions) 
        if (config_.obs_dim == 96 && config_.action_dim == 29) {
            int action_start_idx = 96-29;
            for(int i=0; i<config_.action_dim; ++i) {
                int stat_idx = action_start_idx + i;
                if (stat_idx >= config_.obs_dim) stat_idx = config_.obs_dim - 1;
                
                float val = prev_action[i];
                float norm_val = (val);
                input_vec.push_back(norm_val);
            }
        }
        if (config_.obs_dim == 140 && config_.action_dim == 43) {
            int action_start_idx = 140-43;
            for(int i=0; i<config_.action_dim; ++i) {
                int stat_idx = action_start_idx + i;
                if (stat_idx >= config_.obs_dim) stat_idx = config_.obs_dim - 1;
                
                float val = prev_action[i];
                float norm_val = (val);
                input_vec.push_back(norm_val);
            }
        }
        

        // Run Inference
        auto output = runInference(input_vec);
        result.reconstruction_mu = output.mu;
        result.reconstruction_logvar = output.log_var;

        // Compute Loss vs Current Obs (Target)
        // Normalize Target
        Eigen::VectorXf curr_obs_norm = (curr_obs - obs_mean_).cwiseQuotient(obs_std_);
        
        result.losses.resize(config_.obs_dim);
        result.values.assign(curr_obs_norm.data(), curr_obs_norm.data() + curr_obs_norm.size());

        for (int i = 0; i < config_.obs_dim; i++)
        {
            float mu_val = output.mu[i];
            float log_var_val = output.log_var[i];
            
            // NLL-like anomaly score (drop log-var penalty)
            float precision = std::exp(-log_var_val);
            float diff = curr_obs_norm[i] - mu_val;
            float mse = diff * diff;
            float mse_over_var = precision * mse;
            float median = (i < static_cast<int>(calib_median_mse_over_var_.size())) ? calib_median_mse_over_var_[i] : 1.0f;
            if (median < 1e-6f || !std::isfinite(median)) median = 1.0f;
            float nll = mse_over_var; /// median;
            
            result.losses[i] = nll;
        }

        result.reconstruction_loss_mean = std::accumulate(result.losses.begin(), result.losses.end(), 0.0f) / result.losses.size();
        result.reconstruction_loss_max = *std::max_element(result.losses.begin(), result.losses.end());

        result.current_mean_loss = result.reconstruction_loss_mean;
        result.limit_mean_loss = calib_max_mean_err_; // Now holds Mean+3Sigma
        if (result.current_mean_loss > result.limit_mean_loss) {
            result.is_mean_loss_anomaly = true;
        }

        // Per-Dimension NLL Spike Check
        for (int i = 0; i < config_.obs_dim; ++i) {
             float limit = calib_thresholds_[i];
             
             if (result.losses[i] > limit) {
                 result.is_dim_loss_anomaly = true;
                 result.top_anomaly_indices.push_back(i);
             }
        }
        
        // Range Detector Check (Raw values vs global Min/Max across all timesteps)
        const Eigen::VectorXf& range_min = obs_min_global_;
        const Eigen::VectorXf& range_max = obs_max_global_;
        for(int i=0; i<config_.obs_dim; ++i) {
             float range = range_max[i] - range_min[i];
             if(range < 1e-6f) range = 1.0f;
             float buffer = range * 1.0f + 0.0f;
             if (curr_obs[i] < range_min[i] - buffer || curr_obs[i] > range_max[i] + buffer) {
                 result.is_range_anomaly = true;
                 result.top_anomaly_indices.push_back(i);
             }
        }

        if (result.is_dim_loss_anomaly || result.is_mean_loss_anomaly || result.is_range_anomaly) {
            result.is_anomaly = true;
            for(int idx : result.top_anomaly_indices) {
                // Populate Names
                if(idx < obs_names_.size()) 
                    result.top_anomaly_names.push_back(obs_names_[idx]);
                else 
                    result.top_anomaly_names.push_back("dim_" + std::to_string(idx));
                
                // Populate Limits (Loss) - using new thresholds
                result.top_anomaly_loss_limits.push_back(calib_thresholds_[idx]);

                // Populate calibration median from mse_over_var.csv used for per-dim scaling
                float median = (idx < static_cast<int>(calib_median_mse_over_var_.size()))
                                   ? calib_median_mse_over_var_[idx]
                                   : 1.0f;
                if (median < 1e-6f || !std::isfinite(median)) {
                    median = 1.0f;
                }
                result.top_anomaly_mse_over_var_medians.push_back(median);
                
                // Populate Limits (Range)
                float range = range_max[idx] - range_min[idx];
                if (range < 1e-6f) range = 1.0f;
                float buffer = range * 1.0f + 0.0f;
                result.top_anomaly_range_mins.push_back(range_min[idx] - buffer);
                result.top_anomaly_range_maxs.push_back(range_max[idx] + buffer);
            }
            logAnomaly(result);
        }

        return result;
    }

    void logTransitionSample(const Eigen::VectorXf& obs,
                             const std::vector<float>& action,
                             int timestep)
    {
        if (!config_.enable_logging || !transition_log_file_.is_open())
        {
            return;
        }

        transition_log_file_ << timestep;
        for (int i = 0; i < obs.size(); ++i)
        {
            transition_log_file_ << "," << obs[i];
        }
        for (float value : action)
        {
            transition_log_file_ << "," << value;
        }
        transition_log_file_ << "\n";
        transition_log_file_.flush();
    }

private:
    struct InferenceOutput {
        std::vector<float> per_dim_loss; // NLL per dimension
        std::vector<float> mu;           // Predicted Mean
        std::vector<float> log_var;      // Predicted Log Variance
    };
    void initializeObservationNames()
    {
        obs_names_.clear();
        obs_names_.reserve(config_.obs_dim);

        std::vector<std::string> joint_names = {
        // 29 body joints (IsaacLab order)
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
        };

        if (config_.obs_dim == 140) {
            // 14 hand joints (IsaacLab order)
            std::vector<std::string> hand_joint_names = {
                "left_hand_index_0_joint",
                "left_hand_middle_0_joint",
                "left_hand_thumb_0_joint",
                "right_hand_index_0_joint",
                "right_hand_middle_0_joint",
                "right_hand_thumb_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_1_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_thumb_2_joint",
            };
            joint_names.insert(joint_names.end(), hand_joint_names.begin(), hand_joint_names.end());
        }
        if (config_.obs_dim == 96) {
            // Velocity task
            // --------------------
            // VELOCITY TASK:
            // 1. Linear Velocity (3)
            obs_names_.push_back("root_vel_x");
            obs_names_.push_back("root_vel_y");
            obs_names_.push_back("root_vel_z");

            // 2. Projected Gravity (3) (6)
            obs_names_.push_back("gravity_x");
            obs_names_.push_back("gravity_y");
            obs_names_.push_back("gravity_z");

            // 3. Commands (3)
            obs_names_.push_back("cmd_vel_x");
            obs_names_.push_back("cmd_vel_y");
            obs_names_.push_back("cmd_vel_yaw");
        } else if (config_.obs_dim == 140) { // throwing
            // --------------------
            // THROWING TASK:
            // 1. Linear Velocity (3)
            obs_names_.push_back("root_vel_x");
            obs_names_.push_back("root_vel_y");
            obs_names_.push_back("root_vel_z");

            // 2. Projected Gravity (3) (6)
            obs_names_.push_back("gravity_x");
            obs_names_.push_back("gravity_y");
            obs_names_.push_back("gravity_z");

            // 3. Throwing Target Position (3)
            obs_names_.push_back("target_pos_r");
            obs_names_.push_back("target_pos_theta");
            obs_names_.push_back("target_pos_phi");
        } else {
            // --------------------
            // MIMIC TASK:
            for (const auto& name: joint_names) obs_names_.push_back("ref_pos_" + name);
            for (const auto& name: joint_names) obs_names_.push_back("ref_vel_" + name);
            for (int i = 0; i < 6; ++i) obs_names_.push_back("ref_anchor_rot_6d_" + std::to_string(i));

            obs_names_.push_back("root_ang_vel_x");
            obs_names_.push_back("root_ang_vel_y");
            obs_names_.push_back("root_ang_vel_z");
            // --------------------
        }

        // Safety check
        if (joint_names.size() != 29) {
             std::cerr << "[Warning] Joint name list size (" << joint_names.size() << ") != 29" << std::endl;
        }

        // 4. Joint Positions (29)
        for (const auto& name : joint_names) obs_names_.push_back("pos_" + name);

        // 5. Joint Velocities (29)
        for (const auto& name : joint_names) obs_names_.push_back("vel_" + name);

        // 6. Last Actions (29)
        for (const auto& name : joint_names) obs_names_.push_back("action_" + name);

        if (config_.obs_dim == 140) {
            obs_names_.push_back("estim_displace");
            obs_names_.push_back("not_opened");
        }

        // Final Verification
        if (obs_names_.size() != config_.obs_dim) {
            std::cerr << "[Error] Obs name size (" << obs_names_.size() 
                      << ") mismatches config dim (" << config_.obs_dim << ")" << std::endl;
            // Fallback to avoid crash
            obs_names_.resize(config_.obs_dim, "obs_unknown");
        }
    }

    void loadAEModel()
    {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        try
        {
            ae_session_ = std::make_unique<Ort::Session>(env_, config_.ae_model_path.c_str(), session_options_);
            std::cout << "[AnomalyDetectorFDM] RAPT loaded successfully" << std::endl;
        }
        catch (const Ort::Exception& e)
        {
            throw std::runtime_error("Failed to load RAPT model: " + std::string(e.what()));
        }
    }

    void loadObservationStats()
    {
        try {
            std::cout << "[AnomalyDetectorFDM] Loading stats from H5: " << config_.obs_stats_path << std::endl;

            // 1. Open the HDF5 File
            H5::H5File file(config_.obs_stats_path, H5F_ACC_RDONLY);

            // 2. Helper Lambda to read 1D datasets into Eigen::VectorXf
            auto load_h5_vector = [&](const std::string& name, Eigen::VectorXf& vec) {
                H5::DataSet dataset = file.openDataSet(name);
                H5::DataSpace dataspace = dataset.getSpace();
                int rank = dataspace.getSimpleExtentNdims();
                if (rank != 1) {
                    throw std::runtime_error("Expected 1D dataset for '" + name + "'");
                }
                hsize_t dims_out[1];
                dataspace.getSimpleExtentDims(dims_out, NULL);
                size_t dim = dims_out[0];
                if (dim != static_cast<size_t>(config_.obs_dim)) {
                    throw std::runtime_error("Dimension mismatch in dataset '" + name +
                        "': Expected " + std::to_string(config_.obs_dim) +
                        ", Got " + std::to_string(dim));
                }
                std::vector<float> buffer(dim);
                dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
                vec.resize(dim);
                for(size_t i=0; i<dim; ++i) vec[i] = buffer[i];
            };

            // 3. Helper Lambda to read 1D/2D datasets into Eigen::MatrixXf (min/max per timestep)
            auto load_h5_matrix = [&](const std::string& name, Eigen::MatrixXf& mat) {
                H5::DataSet dataset = file.openDataSet(name);
                H5::DataSpace dataspace = dataset.getSpace();
                int rank = dataspace.getSimpleExtentNdims();
                if (rank != 1 && rank != 2) {
                    throw std::runtime_error("Expected 1D or 2D dataset for '" + name + "'");
                }
                hsize_t dims_out[2] = {0, 0};
                dataspace.getSimpleExtentDims(dims_out, NULL);
                size_t rows = (rank == 2) ? dims_out[0] : 1;
                size_t cols = (rank == 2) ? dims_out[1] : dims_out[0];
                if (cols != static_cast<size_t>(config_.obs_dim)) {
                    throw std::runtime_error("Dimension mismatch in dataset '" + name +
                        "': Expected cols=" + std::to_string(config_.obs_dim) +
                        ", Got cols=" + std::to_string(cols));
                }
                std::vector<float> buffer(rows * cols);
                dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
                mat.resize(static_cast<int>(rows), static_cast<int>(cols));
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t c = 0; c < cols; ++c) {
                        mat(static_cast<int>(r), static_cast<int>(c)) = buffer[r * cols + c];
                    }
                }
            };

            // 4. Load datasets
            load_h5_vector("mean", obs_mean_);
            load_h5_vector("std", obs_std_);
            load_h5_matrix("min", obs_min_ts_);
            load_h5_matrix("max", obs_max_ts_);
            updateGlobalRangeBounds();
            
            std::cout << "[AnomalyDetectorFDM] Loaded H5 stats successfully." << std::endl;

        } catch (const H5::Exception& e) {
            throw std::runtime_error("HDF5 Error loading stats: " + e.getDetailMsg());
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load obs stats (H5): " + std::string(e.what()));
        }
    }

    Eigen::VectorXf getRangeRow(const Eigen::MatrixXf& mat, int timestep) const
    {
        if (mat.rows() == 0) {
            return Eigen::VectorXf::Zero(config_.obs_dim);
        }
        int idx = std::clamp(timestep, 0, static_cast<int>(mat.rows() - 1));
        return mat.row(idx).transpose();
    }

    void updateGlobalRangeBounds()
    {
        if (obs_min_ts_.rows() == 0 || obs_max_ts_.rows() == 0) {
            obs_min_global_ = Eigen::VectorXf::Zero(config_.obs_dim);
            obs_max_global_ = Eigen::VectorXf::Zero(config_.obs_dim);
            return;
        }

        obs_min_global_ = obs_min_ts_.colwise().minCoeff().transpose();
        obs_max_global_ = obs_max_ts_.colwise().maxCoeff().transpose();
    }

    int getClampedRangeRowIndex(int timestep) const
    {
        if (obs_min_ts_.rows() == 0) {
            return 0;
        }
        int mapped = timestep + config_.range_timestep_offset;
        return std::clamp(mapped, 0, static_cast<int>(obs_min_ts_.rows() - 1));
    }

    InferenceOutput runAEInference(const Eigen::VectorXf& obs_normalized)
    {
        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, config_.obs_dim};
        std::vector<float> input_data(obs_normalized.data(), 
                                     obs_normalized.data() + obs_normalized.size());
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"reconstruction"};
        
        auto outputs = ae_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Get reconstruction
        float* reconstruction_data = outputs[0].GetTensorMutableData<float>();
        // includes log_var output
        Eigen::VectorXf reconstruction = Eigen::Map<Eigen::VectorXf>(reconstruction_data, config_.obs_dim*2);
        InferenceOutput out;
        out.per_dim_loss.resize(config_.obs_dim);
        out.mu.resize(config_.obs_dim);
        out.log_var.resize(config_.obs_dim);

        for (int i = 0; i < config_.obs_dim; i++)
        {
            float mu_val = reconstruction[i];
            float log_var_val = reconstruction[i + config_.obs_dim];
            
            // Clamp LogVar
            log_var_val = std::clamp(log_var_val, -6.0f, 6.0f);
            
            out.mu[i] = mu_val;
            out.log_var[i] = log_var_val;

            // Calculate NLL-like anomaly score (drop log-var penalty)
            float precision = std::exp(-log_var_val);
            float diff = obs_normalized[i] - mu_val;
            float mse = diff * diff;

            out.per_dim_loss[i] = precision * mse;
        }

        // Return the struct
        return out;
    }

    void findTopAnomalies(AnomalyResult& result, const Eigen::VectorXf& obs)
    {
        const auto& metric = result.threshold_delta_loss.empty() ? result.losses : result.threshold_delta_loss;
        
        std::vector<std::pair<float, int>> sorted_indices;
        sorted_indices.reserve(metric.size());
        
        for (int i = 0; i < metric.size(); i++) {
            sorted_indices.push_back({metric[i], i});
        }
        
        std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<>());
        
        result.top_anomaly_indices.clear();
        result.top_anomaly_names.clear();
        result.top_anomaly_values.clear();
        
        for (int i = 0; i < std::min(config_.obs_dim, (int)sorted_indices.size()); i++)
        {
            int idx = sorted_indices[i].second;
            result.top_anomaly_indices.push_back(idx);
            if(idx < obs_names_.size())
                result.top_anomaly_names.push_back(obs_names_[idx]);
            else
                result.top_anomaly_names.push_back("dim_" + std::to_string(idx));
            result.top_anomaly_values.push_back(obs[idx]);
        }
    }

    void setupLogging()
    {
        std::string command = "mkdir -p " + config_.log_dir;
        int ret = system(command.c_str());
        (void)ret; 

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << config_.log_dir << "/anomaly_log_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";

        std::stringstream transition_ss;
        transition_ss << config_.log_dir << "/transition_log_"
                      << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
        
        log_file_.open(ss.str());
        log_file_ << "timestep,is_anomaly,z_mean,z_max,loss_mean,loss_max,top_dims,top_names\n";

        transition_log_file_.open(transition_ss.str());
        transition_log_file_ << "timestep";
        for (int i = 0; i < config_.obs_dim; ++i)
        {
            transition_log_file_ << ",obs_" << i;
        }
        for (int i = 0; i < config_.action_dim; ++i)
        {
            transition_log_file_ << ",action_" << i;
        }
        transition_log_file_ << "\n";
    }

    void logAnomaly(const AnomalyResult& result)
    {
        if (!config_.enable_logging || !log_file_.is_open())
        {
            return;
        }
        
        log_file_ << result.timestep << ","
                 << result.is_anomaly << ","
                 << result.reconstruction_loss_mean << ","
                 << result.reconstruction_loss_max << ",\"";
        
        // Log top anomaly dimensions
        for (size_t i = 0; i < result.top_anomaly_indices.size(); i++)
        {
            if (i > 0) log_file_ << ";";
            log_file_ << result.top_anomaly_indices[i];
        }
        log_file_ << "\",\"";
        
        // Log top anomaly names
        for (size_t i = 0; i < result.top_anomaly_names.size(); i++)
        {
            if (i > 0) log_file_ << ";";
            log_file_ << result.top_anomaly_names[i];
        }
        log_file_ << "\"\n";
        log_file_.flush();
    }

    InferenceOutput runInference(const std::vector<float>& input_vec)
    {
        // Dynamic Input Shape: [1, Obs+Act]
        std::vector<int64_t> input_shape = {1, (int64_t)input_vec.size()};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, 
            const_cast<float*>(input_vec.data()), 
            input_vec.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        std::vector<int64_t> hidden_shape = {1, 1, (int64_t)gru_hidden_state_.size()}; // [NumLayers, Batch, HiddenSize]
        Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, 
            gru_hidden_state_.data(), 
            gru_hidden_state_.size(), 
            hidden_shape.data(), 
            hidden_shape.size()
        );

        // 3. Setup Inputs Array
        const char* input_names[] = {"input", "hidden_in"}; 
        const char* output_names[] = {"reconstruction", "hidden_out"}; 
        
        std::vector<Ort::Value> inputs;
        inputs.reserve(2);
        inputs.push_back(std::move(input_tensor));
        inputs.push_back(std::move(hidden_tensor));

        // 4. Run Inference
        auto outputs = ae_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, 
            inputs.data(), // Pass the array of values
            2,             
            output_names, 
            2              
        );
        
        // Update Hidden State (Output index 1)
        float* new_hidden = outputs[1].GetTensorMutableData<float>();
        std::memcpy(gru_hidden_state_.data(), new_hidden, gru_hidden_state_.size() * sizeof(float));
        
        // Process Reconstruction (Output index 0)
        float* raw_out = outputs[0].GetTensorMutableData<float>();
        // Output size is Obs_Dim * 2 (Mu, LogVar)
        
        InferenceOutput out;
        out.mu.resize(config_.obs_dim);
        out.log_var.resize(config_.obs_dim);

        for (int i = 0; i < config_.obs_dim; i++)
        {
            out.mu[i] = raw_out[i];
            float lv = raw_out[i + config_.obs_dim];
            out.log_var[i] = std::clamp(lv, -6.0f, 6.0f);
        }

        return out;
    }
};

} // namespace isaaclab