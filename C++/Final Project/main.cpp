#include <iostream>
#include <vector>
#include <string>
#include "date.h"
#include "termstructure.h"
#include "plainvanilla_option.h"
#include "gbmprocess.h"
#include "binary_option.h"

void print(Option* input_inst, double* ptr) {
	std::vector<double> greek;
	greek.push_back(input_inst->price());
	greek.push_back(input_inst->delta());
	greek.push_back(input_inst->gamma());
	greek.push_back(input_inst->vega());
	greek.push_back(input_inst->rho());
	greek.push_back(input_inst->theta());
	for (unsigned int i = 0; i < greek.size(); ++i)
	{
		printf("%7.2f|  ", greek[i]);
	}
	printf("\n");
	for (unsigned int i = 0; i < greek.size(); ++i)
	{
		ptr[i] += greek[i];
	}
}

void Optionpricing(GBMProcess process, std::vector<Date> mat, std::vector<double> k, std::vector<OptionType> type, Date evalDate, std::vector<std::string> kinds) {
	
	std::vector<Option*> products;
	for (int i = 0; i < 12; ++i) {
		if (kinds[i] == "vanilla") {
			Option* inst = new PlainVanillaOption(mat[i], k[i], type[i]);
			products.push_back(inst);
		}
		else {
			Option* inst = new BinaryOption(mat[i], k[i], type[i]);
			products.push_back(inst);
		}

	}
	for (unsigned int i = 0; i < products.size(); ++i) {
		products[i]->setEvalDate(evalDate);
		products[i]->setProcess(process, evalDate);
	}
	int numgreek = 6;
	double* option_greek = new double[numgreek];

	for (int i = 0; i < numgreek; ++i)
	{
		option_greek[i] = 0;
	}
	printf(" No.  | price |  delta  |  gamma  |   vega  |   rho   |  theta  | \n");


	for (unsigned int i = 0; i < products.size(); ++i) {
		printf(" %2d  |", i + 1);
		print(products[i], option_greek);

	}
	std::cout << std::string(63, '-') << std::endl;
	printf("합계 |%7.2f|%9.2f|%9.2f|%9.2f|%9.2f|%9.2f| \n", option_greek[0], option_greek[1], option_greek[2], option_greek[3], option_greek[4], option_greek[5]);
	for (unsigned int i = 0; i < products.size(); ++i) {
		delete products[i];
	}
	delete[] option_greek;
}


int main() {
	std::vector<Date> dates = { Date(2018,9,30), Date(2018,10,30), Date(2018,11,30), Date(2018,12,30), Date(2019,1,30), Date(2019, 2, 28), Date(2019, 3, 30), Date(2019, 4, 30) };
	Date evalDate(2018, 9, 30);
	double spot = 200;
	std::vector<double> rf = { 0.015, 0.015, 0.017, 0.0185, 0.0195, 0.0205, 0.0213, 0.022 }, div = { 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.04, 0.04 }, vol = { 0.1, 0.11, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145 };
	GBMProcess process(spot, rf, div, vol, dates);
	std::vector<std::string> kinds = { "vanilla","vanilla","vanilla","vanilla","vanilla","vanilla","binary",
		"binary","binary" ,"binary" ,"binary" ,"binary" };
	std::vector<int> position = { 1, -1,1,-1,-1,1,-1,1,-1,1,1,-1 };
	std::vector<OptionType> type = { Call, Call, Call, Put, Put,Put,Call, Call, Put,Put,Put,Call, };
	std::vector<double> k = { 200, 205,195,200,210,190,200,220,200,210,190,205 };
	std::vector<Date> mat = { Date(2019,1,10), Date(2018, 12, 12), Date(2019, 3, 15), Date(2018, 12, 12), Date(2019, 3, 15), Date(2019, 1, 10),
										 Date(2018, 11, 25), Date(2019, 3, 20), Date(2019, 2, 18), Date(2018, 12, 19), Date(2019, 1, 15), Date(2019, 2, 25) };

	std::cout << "< 3번 표에 대한 옵션 가격 및 Greek > \n" << std::endl;
	Optionpricing(process, mat, k, type, evalDate, kinds);
	std::cout << "\n" << "< 주가가 10% 상승한 경우 >" << std::endl;;
	GBMProcess process2(spot*1.1, rf, div, vol, dates);
	Optionpricing(process2, mat, k, type, evalDate, kinds);

	std::cout << "\n" << "< 주가가 10% 하락한 경우 >" << std::endl;;
	GBMProcess process3(spot*0.9, rf, div, vol, dates);
	Optionpricing(process3, mat, k, type, evalDate, kinds);
	printf("\n\n");
	std::vector<Option*> products;
	for (unsigned int i = 0; i < 6; ++i) {
		Option* inst = new PlainVanillaOption(mat[i], k[i], type[i]);
		products.push_back(inst);
	}
	double change_greek[6][6];
	double change_data[6][6];
	for (unsigned int i = 0; i < products.size(); ++i) {
		products[i]->setEvalDate(evalDate);
		products[i]->setProcess(process, evalDate);
		double delta = products[i]->delta();
		double gamma = products[i]->gamma();
		double vega = products[i]->vega();
		double rho = products[i]->rho();
		double ds = spot * 0.01;
		change_greek[i][0] = delta * ds + 0.5 * gamma * ds * ds;
		change_greek[i][1] = delta * -ds + 0.5 * gamma * -ds * -ds;

		double dvol = process.getVol(evalDate)*0.01;
		change_greek[i][2] = vega * dvol;
		change_greek[i][3] = vega * -dvol;

		double dr = 0.001;
		change_greek[i][4] = rho * dr;
		change_greek[i][5] = rho * -dr;

		change_data[i][0] = products[i]->update_price(ds = ds, 0, 0) - products[i]->price();
		change_data[i][1] = products[i]->update_price(ds = -ds, 0, 0) - products[i]->price();
		change_data[i][2] = products[i]->update_price(0, dvol, 0) - products[i]->price();
		change_data[i][3] = products[i]->update_price(0, -dvol, 0) - products[i]->price();
		change_data[i][4] = products[i]->update_price(0, 0, dr) - products[i]->price();
		change_data[i][5] = products[i]->update_price(0, 0, -dr) - products[i]->price();

	}
	
	std::cout << "#Assignment 1\n";
	std::cout << "(1)" << std::endl;
	for (int n = 1; n < 7; ++n) {
		std::cout<<"< " << n << "번째 Plain-vanilla옵션의 가격 변동" << " >"<<std::endl;
		std::cout << std::string(63, '-') << std::endl;
		printf("  시나리오 | 가격 변동 |Greek으로 계산한 가격변동| 오차 \n");
		printf("  주가 +1%% |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][0], change_greek[n - 1][0], change_data[n - 1][0] - change_greek[n - 1][0]);
		printf("  주가 -1%% |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][1], change_greek[n - 1][1], change_data[n - 1][1] - change_greek[n - 1][1]);
		printf("변동성 +1%% |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][2], change_greek[n - 1][2], change_data[n - 1][2] - change_greek[n - 1][2]);
		printf("변동성 -1%% |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][3], change_greek[n - 1][3], change_data[n - 1][3] - change_greek[n - 1][3]);
		printf("금리 +10bp |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][4], change_greek[n - 1][4], change_data[n - 1][4] - change_greek[n - 1][4]);
		printf("금리 -10bp |   %6.3f  |	%10.3f 	 |%6.3f\n", change_data[n - 1][5], change_greek[n - 1][5], change_data[n - 1][5] - change_greek[n - 1][5]);

		std::cout << std::string(63, '-') << std::endl;
	}
	std::cout << "(2)" << std::endl;
	printf("옵션 | 뉴턴랩슨 내재변동성 | Bisection 내재변동성 |뉴턴랩슨 - Bisection \n");
	std::cout << std::string(63, '-') << std::endl;
	for (int i = 0; i < 6; i++)
	{
		double newton = products[i]->Newton_Raphson_IV(products[i]->price());
		double bi = products[i]->bisection_IV(products[i]->price());
		printf(" %d          %5.3f                 %5.3f                %7.3f       \n", i + 1, newton, bi, newton - bi);
	}
	std::cout << std::string(63, '-') << std::endl;

	for (unsigned int i = 0; i < products.size(); ++i) { delete products[i]; }


	return 0;
}
