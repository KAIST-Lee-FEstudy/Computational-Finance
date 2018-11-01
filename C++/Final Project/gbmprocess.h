#pragma once
#include<vector>
#include "date.h"

class GBMProcess
{
public:
	GBMProcess() {}
	GBMProcess(double spot, std::vector<double> rf, std::vector<double> div, std::vector<double> vol, std::vector<Date> dates) :
		spot_(spot), rf_(rf), div_(div), vol_(vol), dates_(dates) {}
	double getSpot() { return spot_; }
	double getRf(Date d);
	double getDiv(Date d);
	double getVol(Date d);

	~GBMProcess() {};
private:
	double spot_;
	std::vector<double> rf_, div_, vol_;
	std::vector<Date> dates_;
};

