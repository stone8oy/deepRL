#include "../include/memoryreplay.hpp"

namespace deepRL {
	void MemoryReplay::addTransition(const Transition& transit){
		if (memorypool.size() == memory_capacity)
			memorypool.pop_front();
		memorypool.push_back(transit);
	}

	std::vector<int> MemoryReplay::sampleTransition(){
                CHECK(memorypool.size()>=batch_size) << "memorypool is not enoough";
	 	std::vector<int> transitions;
	 	transitions.reserve(batch_size);
	 	for (auto i = 0; i < batch_size; ++i) {
    		const auto random_transition_idx =
    		std::uniform_int_distribution<int>(0, memorypool.size() - 1)(random_engine);
    		transitions.push_back(random_transition_idx);
  		}

                return transitions;
	 }

Transition MemoryReplay::getTransitionByIdx(const int idx)
{
   return memorypool[idx];
}


}
