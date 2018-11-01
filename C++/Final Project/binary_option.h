#pragma once
#include "option.h"

class BinaryOption : public Option
{
public:
	BinaryOption(Date expiration, double strike, OptionType type) : // �Ϻθ� overriding, overriding���� �� �θ𿡼� ������
		Option(expiration, strike, type) {}
	double price();

};

enum BarrierType { UpIn, UpOut, DownIn, DownOut };

class BarrierOption : public Option {
public:
	BarrierOption(Date expiration, double strike, OptionType type, double barrier, BarrierType btype) :
		Option(expiration, strike, type), barrier_(barrier), btype_(btype) {}
	double price();
private:
	double barrier_;
	BarrierType btype_;
};