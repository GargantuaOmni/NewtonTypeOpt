#pragma once

#include "Eigen\Dense"
#include "linesearch_more.h"

namespace optim 
{
	template <typename Ftype>
	class Newton : public LineSearch<Ftype>
	{
	public:

		Newton();
		~Newton();

		//void optimize(Ftype & func, VectorXd &x0, double tol = double(1.0e-8), int maxItr = 100);

		VectorXd getOptValue() const;
		VectorXd getGradNorm() const;
		double getFuncMin() const;
		int getItrNum() const;

	protected:

		// iteration number
		int itrNum;

		// minimum value of objective function
		double fMin;

		// optimal solution
		VectorXd xOpt;

		// gradient norm for each iteration
		VectorXd gradNorm;

		//dimension
		int dim;

	};
	// class Newton
	template <typename Ftype>
	class DampingNewton : public Newton<Ftype> {
	public:
		DampingNewton();
		~DampingNewton();
		void optimize(Ftype & func, VectorXd &x0, double tol = double(1.0e-8), int maxItr = 2000);

	};

	template <typename Ftype>
	class Broyden : public Newton<Ftype> {
	public:
		Broyden();
		~Broyden();
		void optimize(Ftype & func, VectorXd &x0, double phi = 1, double tol = double(1.0e-8), int maxItr = 2000);

	};

	template <typename Ftype>
	class SR1Newton : public Newton<Ftype> {
	public:
		SR1Newton();
		~SR1Newton();
		void optimize(Ftype & func, VectorXd &x0, double tol = double(1.0e-8), int maxItr = 2000);

	};

}




