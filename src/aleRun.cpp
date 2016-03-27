#include <cmath>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <boost/program_options.hpp>
//#include "prettyprint.hpp"
#include "../include/aleAgent.hpp"
#include "../include/tools.hpp"

/*
  DeepQLearner(
    const ActionVect& legal_actions,
    double epsilon_start,
    const double epsilon_min,
    const double epsilon_decay,
    const int epsilon_explore_idx,
    const int replay_batch_size,
    const int replay_memory_capacity,
    const int replay_start_size,
    const std::string sampleStrategy,
    const int update_frequency,
    const double discount_factor,
    const std::string solver_param
    ):
*/


using namespace deepRL;
namespace po = boost::program_options;
double EpisodeLearning( ALEInterface& ale, deepRL::DeepQLearner& dqlearner, const bool update);
po::variables_map argmap; 


int main(int argc, char** argv) {

    po::options_description desc("Valid options");
    desc.add_options()
        ("help,h", "Help message")
        ("gpu,u", po::value<bool>()->default_value(false), "Use GPU to brew Caffe")
	("gui,i", po::value<bool>()->default_value(false), "Open a GUI window")
        ("rom,o", po::value<std::string>()->default_value("./roms/breakout.bin"), "Rom file to play")
	("epsilon_start,s", po::value<double>()->default_value(1.0), "Epsilon start value for action selection")
	("epsilon_min,m", po::value<double>()->default_value(0.1), "The final epsilon")
	("epsilon_decay,d", po::value<double>()->default_value(0.98), "Epsilon decay ratio")
	("epsilon_explore_idx,x", po::value<int>()->default_value(100000), "Number of iterations needed for epsilon to reach epsilon_min")
	("replay_memory_capacity,c", po::value<int>()->default_value(50000), "Capacity of replay memory")
	("replay_start_size,r", po::value<int>()->default_value(500), "UEnough amount of transitions to start learning")
	("sampleStrategy,y", po::value<std::string>()->default_value("uniform")->required(), "Sampling strategy for transition")
	("update_frequency,f", po::value<int>()->default_value(1000), "Every #update_frequency set target_net_ = net_")
	("discount_factor,n", po::value<double>()->default_value(0.98), "Discount factor of future rewards (0,1]")
	("solver,v", po::value<std::string>()->default_value("./prototxt/aleSolver.prototxt")->required(), "Solver parameter file (*.prototxt)")
	("skip_frame,p", po::value<int>()->default_value(3), "Number of frames skipped")
	("show_frame,e", po::value<bool>()->default_value(false), "Show the current frame in CUI")
	("model,l", po::value<std::string>()->default_value("")->required(), "Model file to load")
	("evaluate,u", po::value<bool>()->default_value(false), "Evaluation mode: only playing a game, no updates")
	("eval_epsilon,k", po::value<double>()->default_value(0.05), "Epsilon used in evaluate mode")
	("target_q_freq,g", po::value<int>()->default_value(1000), "Taregt_q_net_ update frequency");


           
    po::store(parse_command_line(argc, argv, desc), argmap);
    po::notify(argmap);    

    if (argmap.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
   

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  if (argmap["gpu"].as<bool>()) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale(argmap["gui"].as<bool>());
  std::cout << "Init ale environment ok!" << std::endl;

  // Load the ROM file
  ale.loadROM(argmap["rom"].as<std::string>());

  std::cout << "Loading rom: " << argmap["rom"].as<std::string>() << " ok!" << std::endl;

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  std::cout << "Action space: " << legal_actions.size() << std::endl;

  // deepqlearner
  deepRL::DeepQLearner dqlearner(legal_actions, 
				argmap["epsilon_start"].as<double>(),
				argmap["epsilon_min"].as<double>(),
				argmap["epsilon_decay"].as<double>(),
				argmap["epsilon_explore_idx"].as<int>(),
				deepRL::kMinibatchSize,
				argmap["replay_memory_capacity"].as<int>(),
				argmap["replay_start_size"].as<int>(),
				argmap["sampleStrategy"].as<std::string>(),
				argmap["update_frequency"].as<int>(),
				argmap["discount_factor"].as<double>(),
				argmap["solver"].as<std::string>(),
				argmap["evaluate"].as<bool>(),
                                argmap["eval_epsilon"].as<double>(),
                                argmap["target_q_freq"].as<int>());
  dqlearner.Initialize();

  if (argmap["model"].as<std::string>() !="null") {
    // Just evaluate the given trained model
    dqlearner.LoadPretrainedModel(argmap["model"].as<std::string>());
    std::cout << "Loading " << argmap["model"].as<std::string>() << std::endl;
  }
  //evaluate mode
  if (argmap["evaluate"].as<bool>()) {
    auto total_score = 0.0;
      const auto score =
          EpisodeLearning(ale, dqlearner, false);
      std::cout << "score: " << score << std::endl;
    return 0;
  }

  // learning mode
  for (auto episode = 0;; episode++) {
    const auto score = EpisodeLearning(ale, dqlearner, true);
    std::cout << "Episode: " << episode << " ,Score: " << score << std::endl;
    if (episode % 10 == 0) {
      // After every 10 episodes, evaluate the current strength
      const auto eval_score = EpisodeLearning(ale, dqlearner,false);
      std::cout << "evaluation score: " << eval_score << std::endl;
    }
  }
}

/**
 * one episode learning and return the total score
 */
double EpisodeLearning( ALEInterface& ale, deepRL::DeepQLearner& dqlearner, const bool update) {
  assert(!ale.game_over());
  std::deque<deepRL::FrameDataSp> past_frames;
  //dqlearner.replay_memory_.resetPool();

  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    //std::cout << "frame: " << frame << std::endl;
    const auto current_frame = deepRL::PreprocessScreen(ale.getScreen());
    if (argmap["show_frame"].as<bool>()) {
      std::cout << deepRL::DrawFrame(*current_frame) << std::endl;
    }
    past_frames.push_back(current_frame);
    if (past_frames.size() < deepRL::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      for (auto i = 0; i < argmap["skip_frame"].as<int>() + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
    } else {
      if (past_frames.size() > deepRL::kInputFrameCount) {
        past_frames.pop_front();
      }
      deepRL::InputFrames input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      const auto action = dqlearner.SelectAction(input_frames);
      auto immediate_score = 0.0;

      for (auto i = 0; i < argmap["skip_frame"].as<int>() + 1 && !ale.game_over(); ++i) {
        // Last action is repeated on skipped frames
        immediate_score += ale.act(action);
      }

      total_score += immediate_score;

      //clip reward for robust gradient update
      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      const auto reward =
          immediate_score == 0 ?
              0 :
              immediate_score /= std::abs(immediate_score);

      if (update) {
        // Add the current transition to replay memory
        const auto transition = ale.game_over() ?
            deepRL::Transition(input_frames, action, reward, boost::none) :
            deepRL::Transition(
                input_frames,
                action,
                reward,
                deepRL::PreprocessScreen(ale.getScreen()));
        dqlearner.replay_memory_.addTransition(transition);
	//std::cout << "Memorypool Size: " << dqlearner.replay_memory_.memory_size() << std::endl;
        // If the size of replay memory is enough, update DQN
        if (dqlearner.replay_memory_.memory_size() >= argmap["replay_start_size"].as<int>()
	    and dqlearner.numSteps()%argmap["update_frequency"].as<int>()==0 ) {
          dqlearner.MiniBatchUpdate();
        }
      }
    }
  }
  ale.reset_game();
  return total_score;
}

