//#pragma once
//#include "option.h"
//
//class PlainVanillaOption : public Option {
//public:
//	PlainVanillaOption(Date expiration, double strike, OptionType type) :
//		Option(expiration, strike, type) {}
//	// 오버라이딩한 함수 3개 price, vega, impliedVol
//	double price();
//	double vega();
//	double impliedVol(double mktPrice);
//
//	// 자식 클래스에서 추가로 필요한 기능은 여기 public에 추가 
//private:
//	void setVolatility(double sigma) {
//		sigma_ = sigma;
//	}
//};
#pragma once
#include "option.h"

class PlainVanillaOption : public Option {
public:
	PlainVanillaOption(Date expiration, double strike, OptionType type) :
		Option(expiration, strike, type) {}
	double price();
	virtual double delta();
	virtual double gamma();
	virtual double vega();
	virtual double rho();
	virtual double theta();
	double Newton_Raphson_IV(double mktPrice);
	virtual double update_price(double ds = 0, double dvol = 0, double dr = 0);
	virtual double bisection_IV(double mktPrice);
private:
	void setVolatility(double sigma) {
		sigma_ = sigma;
	}
};