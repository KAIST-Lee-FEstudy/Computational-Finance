#include "option.h"
#include <cmath>

void Option::setEvalDate(Date d) {
	evalDate_ = d;
	t_ = daysBetween(evalDate_, expiration_) / 365.0;

}

void Option::change_Inputparams(double ds, double dvol, double dr) {
	s_ += ds;
	sigma_ += dvol;
	r_ += dr;
}

void Option::setProcess(GBMProcess p, Date d) {
	p_ = p;
	s_ = p_.getSpot();
	r_ = p_.getRf(d);
	q_ = p_.getDiv(d);
	sigma_ = p_.getVol(d);
}

double Option::getd1() {
	return (log(s_ / strike_) + (r_ - q_ + 0.5*sigma_*sigma_)*t_) / (sigma_*sqrt(t_));
}
double Option::getd2() {
	return getd1() - sigma_ * sqrt(t_);
}

double Option::delta() {
	s_ *= 1.01;
	double p_1 = price();
	s_ *= 0.99 / 1.01;
	double p_2 = price();
	s_ /= 0.99;
	return (p_1 - p_2) / (0.02*s_);
}
double Option::gamma() {
	double p_0 = price();
	s_ *= 1.01;
	double p_1 = price();
	s_ *= 0.99 / 1.01;
	double p_2 = price();
	s_ /= 0.99;
	return (p_1 - 2 * p_0 + p_2) / ((0.01*s_) * (0.01*s_));
}
double Option::vega() {
	sigma_ += 0.01;
	double p_1 = price();
	sigma_ -= 0.02;
	double p_2 = price();
	sigma_ += 0.01;
	return (p_1 - p_2) / 0.02;
}

double Option::rho() {
	r_ += 0.0001;
	double p_1 = price();
	r_ -= 0.0002;
	double p_2 = price();
	r_ += 0.0001;
	return (p_1 - p_2) / 0.0002;
}

double Option::theta() {
	double p_0 = price();
	t_ += 0.0001;
	double p_1 = price();

	return (p_1 - p_0) / 0.0001;
}