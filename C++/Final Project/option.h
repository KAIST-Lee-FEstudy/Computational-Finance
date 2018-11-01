#pragma once

#include "date.h"
#include "gbmprocess.h"
enum OptionType { Call = 1, Put = -1 };

class Option { 
public:
	Option(Date expiration, double strike, OptionType type) :
		expiration_(expiration), strike_(strike), type_(type) {}
	void setProcess(GBMProcess p, Date d);
	void setEvalDate(Date d);
	void change_Inputparams(double ds = 0, double dvol = 0, double dr = 0);
	virtual double update_price(double ds = 0, double dvol = 0, double dr = 0) { return 0; };
	virtual double price() = 0; 
	virtual double delta();
	virtual double gamma();
	virtual double vega(); 
	virtual double rho();
	virtual double theta();
	virtual double Newton_Raphson_IV(double m) { return 10; }
	virtual double bisection_IV(double mktPrice) { return 10; }
	double s_, r_, q_, t_, sigma_;
protected:
	double getd1();
	double getd2();

	Date evalDate_;
	Date expiration_;
	double strike_;
	OptionType type_;
	GBMProcess p_;

};


