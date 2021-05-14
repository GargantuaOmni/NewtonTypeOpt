#pragma once

#include "Eigen\Dense"
#include "linesearch_more.h"

#define FR 0
#define PRP 1
#define FR_PRP 2
#define FR_H_PRP 3

namespace optim 
{
	template <typename Ftype>
	class Gradient : public LineSearch<Ftype>
	{
	public:

		Gradient();
		~Gradient();

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

		// dimension
		int dim;

	};

	// class Newton
	template <typename Ftype>
	class BB : public Gradient<Ftype> {
	public:
		BB();
		~BB();
		void optimize(Ftype & func, VectorXd &x0, double tol = double(1.0e-6), int maxItr = 20000);

	};

	template <typename Ftype>
	class CG : public Gradient<Ftype> {
	public:
		CG();
		~CG();
		void optimize(Ftype & func, VectorXd &x0, int type, int N, double tol = double(1.0e-6), int maxItr = 20000, double alpha_0 = 1);

	};

}


#pragma once
