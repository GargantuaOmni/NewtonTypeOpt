#pragma once

#include "Eigen\Dense"
#include "constant.h"

using namespace Eigen;

namespace optim
{
	
	template <typename Ftype> 
	class LineSearch
	{
	protected:
		int     funcNum;
		bool    success;
		double   rho;
		double   sigma;


		//NSLA:
		double C, C_next;
		double Q, Q_next;
		double delta;
		double yita;

		//Save:
		double lambda_saved;

	public:

		LineSearch();
		~LineSearch();
		int getFuncNum() const;
		bool isSuccess() const;

		void setRho(double _rho);
		void setSigma(double _sigma);

		//NSLA:
		void setYita(double _yita);
		void setDelta(double _delta);
		void setC(double _C);

		double getStep(Ftype &func, VectorXd &xk, VectorXd &dk,
			int maxItr = 15, short criterion = CRITERION_WOLF, double alpha_0 = 1);

		double getNlsaStep(
			Ftype &func, VectorXd &xk, VectorXd &dk,
			int maxItr = 15, short criterion = CRITERION_MODIFIED_WOLF);

		double CostStep(
			double fx_next
		);

		bool getCriterion(double alpha, double gd, double fx, double fnew, short criterion = CRITERION_WOLF, double gd_new = (double)0);

	

	};
	// class LineSearch



}
