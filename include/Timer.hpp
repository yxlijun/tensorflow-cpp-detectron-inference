#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>

class Timer {
	using Clock = std::chrono::high_resolution_clock;
public:
	inline void Tic() {
		_start = Clock::now();
	}

	inline void Toc() {
		_end = Clock::now();
	}

	inline double Elasped() {
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start);
		return duration.count();
	}

private:
	Clock::time_point _start, _end;
};
#endif