
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
      1,
      kInputFrameCount,
      kCroppedFrameSize,
      kCroppedFrameSize));

  // Cache pointers to target input layers
  target_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("target_input_layer"));
  assert(target_input_layer_);
  assert(deepRL::HasBlobSize(
      *net_->blob_by_name("target"), 1, kOutputCount, 1, 1));

  // Cache pointers to filter input layers
  filter_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("filter_input_layer"));
  assert(filter_input_layer_);
  assert(deepRL::HasBlobSize(
      *net_->blob_by_name("filter"), 1, kOutputCount, 1, 1));
}

//one inputframe mode
std::vector<float>  DeepQLearner::ForwardQvalue(const InputFrames& frames_input,const NetSp& qnet){
  
  std::vector<float> q_values(legal_actions_.size());
  std::array<float,kInputDataSize> frames_input_data;
  for (auto j = 0; j < kInputFrameCount; ++j) {//
    const auto& frame_data = frames_input[j];
    std::copy(
        frame_data->begin(),
        frame_data->end(),
        frames_input_data.begin() + j*kCroppedFrameDataSize);
  }
  FillData2Layers(frames_input_data, dummy_input_data_, dummy_input_data_);

  qnet->Forward();//test ,now we got the qvalue


  // Get the Q values from the net
  const auto action_evaluator = [&](Action action) {
    const auto q = q_values_blob_->data_at(0, static_cast<int>(action), 0, 0);
    //std::cout << q << std::endl;
    assert(!std::isnan(q));
    return q;
  };

  std::transform(
      legal_actions_.begin(),
      legal_actions_.end(),
      q_values.begin(),
      action_evaluator);
  //std::cout << PrintQValues(q_values,legal_actions_) << std::endl;
  return q_values;
}

ActionPair DeepQLearner::MaxActionQvalue(std::vector<float> q_values){
    const auto max_idx =
        std::distance(
            q_values.begin(),
            std::max_element(q_values.begin(), q_values.end()));

    return std::make_pair(legal_actions_[max_idx],q_values[max_idx]);
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
    action =  MaxActionQvalue(ForwardQvalue(input_frames,net_)).first;// max 
    //std::cout << action_to_string(action) << " (greedy)";
  }
  return action;
}

void DeepQLearner::StepUpdate(const Transition& tr){//only using single inputframe
 if (current_iter_%100 ==0)
      std::cout << "iteration: " << current_iter_ << " , epsilon: " <<  epsilon_ << std::endl;
   current_iter_++;


   //extract <s,a,r,s'>
   InputFrames s = std::get<0>(tr);
   const auto a = std::get<1>(tr); 
   const auto r = std::get<2>(tr); assert(r >= -1.0 && r <= 1.0);
   InputFrames s_;
   const auto terminal = std::get<3>(tr) ? false : true;
   if(!terminal) {
    for (auto i = 0; i < kInputFrameCount - 1; ++i) {
      s_[i] = std::get<0>(tr)[i + 1];
    }
    s_[kInputFrameCount - 1] = std::get<3>(tr).get();
  }


  const auto target_q = terminal ?//r+maxQ'(s,a)
          r:
          r + gamma * MaxActionQvalue(ForwardQvalue(s_,target_net_)).second;
  assert(!std::isnan(target_q));


  //do data feeding -> train Q-network
  FramesLayerInputData frames_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  
  // only the changed q has the loss
  target_input[static_cast<int>(a)] = target_q;
  filter_input[static_cast<int>(a)] = 1;//one-hot encoder

    for (auto j = 0; j < kInputFrameCount; ++j) {
      const auto& frame_data = s[j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + j * kCroppedFrameDataSize);
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

void DeepQLearner::BatchUpdate() {
  
  // Sample transitions from replay memory
  //std::cout << "==>begin sampling experiences" << std::endl;
  std::vector<int> transitions = replay_memory_.sampleTransition();
  
  CHECK(transitions.size() == kMinibatchSize) << "Exeperience is not sampled enough";
  
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_.getTransitionByIdx(transitions[i]);
    StepUpdate(transition);
  }
}

void DeepQLearner::FillData2Layers(
      const FramesLayerInputData& frames_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {

    frames_input_layer_->Reset(
      const_cast<float*>(frames_input.data()),//data
      dummy_input_data_.data(),//label
      1);//size n

    target_input_layer_->Reset(
      const_cast<float*>(target_input.data()),
      dummy_input_data_.data(),
      1);

    filter_input_layer_->Reset(
      const_cast<float*>(filter_input.data()),
      dummy_input_data_.data(),
      1);
}

}

