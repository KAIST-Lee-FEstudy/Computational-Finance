#pragma once
#include <iostream>
#include <vector>
#include "date.h"

class TermStructure {
private:
	std::vector<Date> dates_;
	std::vector<double> values_;

protected:
	double timeFromRef(Date d) {
		return daysBetween(dates_[0], d);  //Time convention 365
	}
public:
	TermStructure(std::vector<Date> dates, std::vector<double> values) :
		dates_(dates), values_(values) {}
	~TermStructure() {}
	virtual double value(Date d);
};

class YieldTermStructure : public TermStructure {
public:
	YieldTermStructure(std::vector<Date> date, std::vector<double> rates) :
		TermStructure(date, rates) {}
	double discount_factor(Date d);
	double forwardRate(Date d1, Date d2);
};

class VolTermStructure : public TermStructure {
public:
	VolTermStructure(std::vector<Date> date, std::vector<double> vol) :
		TermStructure(date, vol) {}
	double totalVariance(Date d);
	double value(Date d) { return 0.1; }
};