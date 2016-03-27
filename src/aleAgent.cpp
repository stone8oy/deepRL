
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

#include "../include/tools.hpp"
#include "../include/aleAgent.hpp"


namespace deepRL {

void DeepQLearner::LoadPretrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
}

void DeepQLearner::Initialize() {
  // Initialize net and solver

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
  solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));


  //current q-net
  net_ = solver_->net();

  //target q-net
  target_net_ = solver_->net();

  // Cache pointers to blobs that hold Q values
  q_values_blob_ = net_->blob_by_name("q_values");

  // Initialize dummy input data with 0
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);

  // Cache pointers to input layers
  frames_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("frames_input_layer"));
  assert(frames_input_layer_);
  assert(deepRL::HasBlobSize(
      *net_->blob_by_name("frames"),
      kMinibatchSize,
      kInputFrameCount,
      kCroppedFrameSize,
      kCroppedFrameSize));

  // Cache pointers to target input layers
  target_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("target_input_layer"));
  assert(target_input_layer_);
  assert(deepRL::HasBlobSize(
      *net_->blob_by_name("target"), kMinibatchSize, kOutputCount, 1, 1));

  // Cache pointers to filter input layers
  filter_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("filter_input_layer"));
  assert(filter_input_layer_);
  assert(deepRL::HasBlobSize(
      *net_->blob_by_name("filter"), kMinibatchSize, kOutputCount, 1, 1));
}

ActionPairVec DeepQLearner::ForwardBatchMaxQvalue(const InputFramesVec& batch_frames,const NetSp& qnet){
//(inputframe1,inputframe2,...,inputframe_batch_size)
  assert(batch_frames.size() <= kMinibatchSize);
  std::array<float, kMinibatchDataSize> batch_frames_input;

  for (auto i = 0; i < batch_frames.size(); ++i) {
    // Input frames to the net and compute Q values for each legal actions
    for (auto j = 0; j < kInputFrameCount; ++j) {//
      const auto& frame_data = batch_frames[i][j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          batch_frames_input.begin() + i * kInputDataSize + j * kCroppedFrameDataSize);
    }
  }

  FillData2Layers(batch_frames_input, dummy_input_data_, dummy_input_data_);
  qnet->Forward();//test ,now we got the qvalue

  ActionPairVec results;
  results.reserve(batch_frames.size());
  for (auto i = 0; i < batch_frames.size(); ++i) {
    // Get the Q values from the net
    const auto action_evaluator = [&](Action action) {
      const auto q = q_values_blob_->data_at(i, static_cast<int>(action), 0, 0);
      assert(!std::isnan(q));
      return q;
    };

    std::vector<float> q_values(legal_actions_.size());
    std::transform(
        legal_actions_.begin(),
        legal_actions_.end(),
        q_values.begin(),
        action_evaluator);

    // Select the action with the maximum Q value
    const auto max_idx =
        std::distance(
            q_values.begin(),
            std::max_element(q_values.begin(), q_values.end()));

    results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
  }
  return results;
}



Action DeepQLearner::SelectAction(const InputFrames& last_frames) {
  numSteps_++;
  //update target_net_ with frequency param: target_q_freq
  if ( target_q_freq >0 and numSteps_%target_q_freq ==0){
        target_net_ = solver_->net(); // can we do like this ?
  } 
  assert(epsilon_ >= 0.0 && epsilon_ <= 1.0);
  Action action;
  double valid_epsilon_ = epsilon_;//default TRAIN
  if ( evaluate )  valid_epsilon_ = eval_epsilon;
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < valid_epsilon_) {//random
    const auto random_idx =
        std::uniform_int_distribution<int>(0, legal_actions_.size() - 1)(random_engine);
    action = legal_actions_[random_idx];
    //std::cout << action_to_string(action) << " (random)";
  } else {//max greedy
    action = ForwardBatchMaxQvalue(InputFramesVec{{last_frames}},net_).front().first;// max 
    //std::cout << action_to_string(action) << " (greedy)";
  }
  return action;
}

void DeepQLearner::MiniBatchUpdate() {
  if (current_iter_%1000 ==0)
      std::cout << "iteration: " << current_iter_++ << std::endl;

  // Sample transitions from replay memory
  //std::cout << "==>begin sampling experiences" << std::endl;
  std::vector<int> transitions = replay_memory_.sampleTransition();
  
  CHECK(transitions.size() == kMinibatchSize) << "Exeperience is not sampled enough";
   
  //std::cout << "<==end sampling experiences" << std::endl;
  // for each transition <s,a,r,s'> ,compute the q values for s (net_)and s'(target_net_)

  // Compute target values: max_a Q(s',a)
  InputFramesVec target_last_frames_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_.getTransitionByIdx(idx);
    if (!std::get<3>(transition)) {
      // This is a terminal state
      continue;
    }
    // Compute target value
    InputFrames target_last_frames;
    for (auto i = 0; i < kInputFrameCount - 1; ++i) {
      target_last_frames[i] = std::get<0>(transition)[i + 1];
    }
    target_last_frames[kInputFrameCount - 1] = std::get<3>(transition).get();
    target_last_frames_batch.push_back(target_last_frames);
  }
  //get qvalue from target_net_
  const auto actions_and_values = ForwardBatchMaxQvalue(target_last_frames_batch,target_net_);
 
  //do data feeding -> train Q-network
  FramesLayerInputData frames_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  auto target_value_idx = 0;
  //std::cout << "filling data" << std::endl;
  for (auto i = 0; i < kMinibatchSize; ++i) {

    const auto& transition = replay_memory_.getTransitionByIdx(transitions[i]);

    const auto action = std::get<1>(transition);//action 
    // std::cout << action << std::endl;
    assert(static_cast<int>(action) < kOutputCount);

    const auto reward = std::get<2>(transition);//reward
    assert(reward >= -1.0 && reward <= 1.0);

    const auto target = std::get<3>(transition) ?//r+maxQ'(s,a)
          reward + gamma * actions_and_values[target_value_idx++].second :
          reward;
    assert(!std::isnan(target));

    target_input[i * kOutputCount + static_cast<int>(action)] = target;
    filter_input[i * kOutputCount + static_cast<int>(action)] = 1;


    for (auto j = 0; j < kInputFrameCount; ++j) {
      const auto& frame_data = std::get<0>(transition)[j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize + j * kCroppedFrameDataSize);
    }

  }
  FillData2Layers(frames_input, target_input, filter_input);
 
  solver_->Step(1);


    //epsilon decay
  //std::cout << " epsilon:" << epsilon_ << std::endl;
  if (epsilon_decay_ > 0.0) {
    if (epsilon_ != epsilon_min_){
      epsilon_ = epsilon_*epsilon_decay_;
      if (current_iter_ > epsilon_explore_idx_)
          epsilon_ = epsilon_ < epsilon_min_ ? epsilon_min_ : epsilon_;
    }
  } 


  }

void DeepQLearner::FillData2Layers(
      const FramesLayerInputData& frames_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {

    frames_input_layer_->Reset(
      const_cast<float*>(frames_input.data()),//data
      dummy_input_data_.data(),//label
      kMinibatchSize);//size n

    target_input_layer_->Reset(
      const_cast<float*>(target_input.data()),
      dummy_input_data_.data(),
      kMinibatchSize);

    filter_input_layer_->Reset(
      const_cast<float*>(filter_input.data()),
      dummy_input_data_.data(),
      kMinibatchSize);
}

}

