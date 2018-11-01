#include "gbmprocess.h"
#include "termstructure.h"



double GBMProcess::getRf(Date d) {
	YieldTermStructure ts(dates_, rf_);
	double v = ts.value(d);
	return v;
}

double GBMProcess::getDiv(Date d) {
	YieldTermStructure ts(dates_, div_);
	double v = ts.value(d);
	return v;
}

double GBMProcess::getVol(Date d) {
	YieldTermStructure ts(dates_, vol_);
	double v = ts.value(d);
	return v;
}