#ifndef MEMORYREPLAY_HPP_
#define MEMORYREPLAY_HPP_

#include <deque>
#include "typedata.hpp"

namespace deepRL {
  class MemoryReplay{
  public:
    MemoryReplay(
      int memory_size_,
      int sample_size,
      std::string sampleStrategy_
      ):
    memory_capacity(memory_size_),
    batch_size(sample_size),
    sampleStrategy(sampleStrategy_),
    random_engine(0){}

  public:
    void addTransition(const Transition& transit);
    int replay_memory_capacity_(){return memory_capacity;}
    Transition getTransitionByIdx(const int idx);
    int memory_size(){return memorypool.size();}
    std::vector<int> sampleTransition();
    inline void resetPool(){memorypool.clear();}

  private:
    int memory_capacity;
    std::string sampleStrategy;
    int batch_size;
    std::deque<Transition> memorypool;
    std::mt19937 random_engine;

  };

    

}

#endif /* MEMORYREPLAY_HPP_ */
