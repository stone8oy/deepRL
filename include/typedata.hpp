#ifndef TYPEDATA_HPP_
#define TYPEDATA_HPP_

#include <tuple>
#include <vector>
#include <ale_interface.hpp>
#include <boost/optional.hpp>
#include <boost/functional/hash.hpp>
#include <caffe/caffe.hpp>

namespace deepRL {


//original frame size
constexpr auto kRawFrameHeight = 210;
constexpr auto kRawFrameWidth = 160;
//cropped size
constexpr auto kCroppedFrameSize = 84;
constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
//4 frames -> input
constexpr auto kInputFrameCount = 4;
constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
//batch size
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
//discrete actions
constexpr auto kOutputCount = 4;


using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
using FrameDataSp = std::shared_ptr<FrameData>;
using InputFrames = std::array<FrameDataSp, 4>;
using Transition = std::tuple<InputFrames, Action, float, boost::optional<FrameDataSp>>;//experience

using FramesLayerInputData = std::array<float, kCroppedFrameDataSize>;//caffe input 
using TargetLayerInputData = std::array<float, 1 * kOutputCount>;//Q_target
using FilterLayerInputData = std::array<float, 1 * kOutputCount>;//

using SolverSp = boost::shared_ptr<caffe::Solver<float>>;
using NetSp = boost::shared_ptr<caffe::Net<float>>;
using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;

using ActionPair = std::pair<Action, float>;
using ActionPairVec = std::vector<ActionPair>;
using InputFramesVec = std::vector<InputFrames>;



}


#endif /* TYPEDATA_HPP_ */

