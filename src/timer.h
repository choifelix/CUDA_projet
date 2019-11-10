#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef CLOCKS_PER_SEC
#include <unistd.h>
#define CLOCKS_PER_SEC_SC_CLK_TCK
#endif

class Timer {

private:
	double _start;
	double _sum;

public:
	Timer(void) :  _start(0.0), _sum(0.0){}

public:
	inline void start(void){
		_start = clock();
	}

inline void add(void) {
	_sum += (clock() - _start)/(double)CLOCKS_PER_SEC;
}

inline double getstart(void) const {
	return _start;
}

inline double getsum(void) const {
	return _sum;
}

inline void reset(void){
	_sum=0.0;
};





};