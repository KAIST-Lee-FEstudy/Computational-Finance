#include <cmath>
#include "termstructure.h"

double TermStructure::value(Date d) {
	double t = timeFromRef(d);
	double t1, t2;
	int i = 0;
	for (i = 0; i < dates_.size() - 1; ++i) {
		t1 = timeFromRef(dates_[i]);
		t2 = timeFromRef(dates_[i + 1]);
		if (t >= t1 && t < t2)
			break;
	}
	double v = values_[i] + (values_[i + 1] - values_[i])*(t - t1) / (t2 - t1);  //선형보간 함수 
	return v;
}

double YieldTermStructure::discount_factor(Date d) {
	return exp(-value(d) * timeFromRef(d));
}

double YieldTermStructure::forwardRate(Date d1, Date d2) {
	return 365*(discount_factor(d1) / discount_factor(d2) - 1) / (timeFromRef(d2) - timeFromRef(d1));
}

double VolTermStructure::totalVariance(Date d) {
	return timeFromRef(d) * value(d) * value(d);
}