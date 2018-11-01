#include "plainvanilla_option.h"
#include "normal.h"
#include <cmath>


double PlainVanillaOption::price() {
	double d1 = getd1();
	double d2 = getd2();
	double nd1 = normcdf(type_*d1);
	double nd2 = normcdf(type_*d2);
	double price = type_ * (s_*exp(-q_ * t_)*nd1 - strike_ * exp(-r_ * t_)*nd2);
	return price;
}

double PlainVanillaOption::delta() {
	double d1 = getd1();
	double ncd1 = normcdf(type_*d1);
	double delta = type_ * exp(-q_ * t_)*ncd1;
	return delta;
}

double PlainVanillaOption::gamma() {
	double d1 = getd1();
	double npd1 = normpdf(d1);
	double gamma = exp(-q_ * t_)*npd1 / (s_*sigma_*sqrt(t_));
	return gamma;
}

double PlainVanillaOption::vega() {
	double d1 = getd1();
	double npd1 = normpdf(d1);
	double vega = s_ * exp(-q_ * t_)*npd1*sqrt(t_);
	return vega;
}

double PlainVanillaOption::rho() {
	double d2 = getd2();
	double ncd2 = normcdf(type_ * d2);
	double rho = type_ * strike_ * t_ * exp(-r_ * t_)*ncd2;
	return rho;
}

double PlainVanillaOption::theta() {
	double d1 = getd1();
	double d2 = getd2();
	double npd1 = normpdf(d1);
	double ncd1 = normcdf(type_ * d1);
	double ncd2 = normcdf(type_ * d2);
	double theta = -exp(-q_ * t_) * s_ * npd1 * sigma_ / (2 * sqrt(t_)) + -type_ * r_ * strike_ * exp(-r_ * t_)*ncd2 + type_ * q_ * s_ * exp(-q_ * t_) * ncd1;
	return theta;
}

double PlainVanillaOption::Newton_Raphson_IV(double mktPrice) {
	double init = 0.1;
	double tol = 1e-8;
	double x = init;
	double e = 1;
	while (e > tol) {
		setVolatility(x);
		double diff = price() - mktPrice;
		e = abs(diff);
		x = x - diff / vega();
	}
	return x;
}

double PlainVanillaOption::update_price(double ds, double dvol, double dr)
{
	s_ += ds;
	sigma_ += dvol;
	r_ += dr;
	double d1 = getd1();
	double d2 = getd2();
	double nd1 = normcdf(type_*d1);
	double nd2 = normcdf(type_*d2);
	double price = type_ * (s_*exp(-q_ * t_)*nd1 - strike_ * exp(-r_ * t_)*nd2);

	s_ -= ds;
	sigma_ -= dvol;
	r_ -= dr;

	return price;
}

double PlainVanillaOption::bisection_IV(double mktPrice)
{
	double tol = 1e-8;
	double lower = 0;
	double upper = 1;
	double middle_point = 0.5;
	double e = 1;


	do
	{
		setVolatility(middle_point);


		e = price() - mktPrice;
		if (e < 0) {
			lower = middle_point;
		}
		else {
			upper = middle_point;
		}
		middle_point = (lower + upper) / 2;

	} while (abs(e) > tol);

	return middle_point;
}

