//#pragma once
//#include "option.h"
//
//class PlainVanillaOption : public Option {
//public:
//	PlainVanillaOption(Date expiration, double strike, OptionType type) :
//		Option(expiration, strike, type) {}
//	// �������̵��� �Լ� 3�� price, vega, impliedVol
//	double price();
//	double vega();
//	double impliedVol(double mktPrice);
//
//	// �ڽ� Ŭ�������� �߰��� �ʿ��� ����� ���� public�� �߰� 
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