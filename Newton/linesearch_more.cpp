#include "linesearch_more.h"
#include "cmath"
#include <iostream>

using namespace Eigen;
namespace optim 
{
	template <typename Ftype>
	LineSearch<Ftype>::LineSearch()
	{
		funcNum = 0;
		Q = 1; // Only for NLSA (Nonmonotone Line Search Algorithm)
		lambda_saved = 0;
		success = true;
	}

	template <typename Ftype>
	LineSearch<Ftype>::~LineSearch()
	{
	}


	/**
	 * Get the number of objective function's calculation.
	 */
	template <typename Ftype>
	inline int LineSearch<Ftype>::getFuncNum() const
	{
		return funcNum;
	}


	/**
	 * Judgement whether the optimal solution is found or not.
	 */
	template <typename Ftype>
	inline bool LineSearch<Ftype>::isSuccess() const
	{
		return success;
	}

	template <typename Ftype>
	void LineSearch<Ftype>::setRho(double _rho) {
		rho = _rho;
		return;
	}

	template <typename Ftype>
	void LineSearch<Ftype>::setSigma(double _sigma) {
		sigma = _sigma;
		return;
	}

	template <typename Ftype>
	void LineSearch<Ftype>::setYita(double _yita) {
		yita = _yita;
		return;
	}

	template <typename Ftype>
	void LineSearch<Ftype>::setDelta(double _delta) {
		delta = _delta;
		return;
	}

	template <typename Ftype>
	void LineSearch<Ftype>::setC(double _C) {
		C = _C;
		return;
	}

	template <typename Ftype>
	inline bool LineSearch<Ftype>::getCriterion(double alpha, double gd, double fx, double fnew, short c, double gd_new) {
		return (c == CRITERION_ARMIJO && fnew <= fx + rho * gd *  alpha)
			|| (c == CRITERION_GOLDSTEIN && fnew <= fx + rho * gd *  alpha && fnew >= fx + (1 - rho) * gd * alpha)
			|| (c == CRITERION_WOLF && fnew <= fx + rho * gd *  alpha && gd_new >= sigma * gd)
			|| (c == CRITERION_STRONG_WOLF && fnew <= fx + rho * gd *  alpha && fabs(gd_new) <= -1 * sigma * gd)
			|| (c == CRITERION_MODIFIED_ARMIJO && fnew <= C + rho * gd * alpha) 
			|| (c == CRITERION_MODIFIED_WOLF && fnew <= C + rho *gd * alpha  && gd_new >= sigma * gd);
	}

	template <typename Ftype>
	double LineSearch<Ftype>::getStep(Ftype &func, VectorXd &xk, VectorXd &dk, int maxItr, short criterion, double alpha_0)
	{
		// Set line search parameters that everyone uses.
		double kUp = double(0.5),
			kLow = double(0.1),
			alpha = alpha_0,
			alphaMin,
			alphaMax;
		

		double gdNew = 0;

		std::cout << "Debug point linesearch" << std::endl;

		double fNew,
			fk = func(xk);

		funcNum++;

		VectorXd xNew,
			gk = func.grad(xk);

		double gd = gk.dot(dk);

		std::cout << "Debug gd:" << gd << std::endl;

		VectorXd gNew;
		//std::cout << "linesearch fx:" << fk << std::endl;
		for (int i = 0; i < maxItr; ++i)
		{
			xNew = xk + alpha * dk;
			fNew = func(xNew);
			std::cout << "dk:" << dk.norm() <<  "linesearch fNew:" << fNew << " alpha=" <<  alpha << std::endl;
			if (criterion == CRITERION_WOLF || criterion == CRITERION_STRONG_WOLF)
			{
				gNew = func.grad(xNew);
				gdNew = gNew.dot(dk);
			}
				
			funcNum++;

			if (getCriterion(alpha, gd, fk, fNew, criterion, gdNew))
			{
				success = true;
				//std::cout << "Linesearch secceeds: " << alpha << std::endl;
				return alpha;
			}
			else
			{
				alphaMin = kLow * alpha;
				alphaMax = kUp * alpha;

				// Compute the step by using quadratic polynomial interpolation.
				alpha = double(-0.5)*alpha*alpha*gd / (fNew - fk - alpha * gd + EPS);

				// bound checking
				if (alpha < alphaMin || isnan(alpha))
					alpha = alphaMin;
				else if (alpha > alphaMax)
					alpha = alphaMax;
			}
		}
		success = false;
		std::cout << "Linesearch 15 times..." << std::endl;
		if (fNew >= fk)
		{
			success = false;
			return alpha;
			
		}
		else
		{
			success = true;
			std::cout << "Linesearch ends: " << alpha << std::endl;
			return alpha;

		}
	}

	template <typename Ftype>
	double LineSearch<Ftype>::getNlsaStep(Ftype &func, VectorXd &xk, VectorXd &dk, int maxItr, short criterion)
	{
		// Set line search parameters that everyone uses.
		double alpha;

		double kUp = double(0.5),
			kLow = double(0.1),
			alphaMin,
			alphaMax;

		if (lambda_saved < EPSS || lambda_saved > 1.0 / EPSS) alpha = 10;
		else alpha = 1.0 / lambda_saved;

		double gdNew = 0;

		//std::cout << "Debug point linesearch (NLSA)" << std::endl;

		double fNew,
			fk = func(xk);

		funcNum++;

		VectorXd xNew, 
			gk = func.grad(xk);

		double gd = gk.dot(dk);

		std::cout << "Debug gd:" << gd << std::endl;

		VectorXd gNew, y;
		//std::cout << "linesearch fx:" << fk << std::endl;
		for (int i = 0; i < maxItr; ++i)
		{
			xNew = xk + alpha * dk;
			fNew = func(xNew);
			std::cout << "dk:" << dk.norm() << "linesearch fNew:" << fNew << " alpha=" << alpha << std::endl;
			if (criterion == CRITERION_MODIFIED_WOLF )
			{
				gNew = func.grad(xNew);
				gdNew = gNew.dot(dk);
			}

			funcNum++;
			//if (alpha < 1e-8)break;
			if (getCriterion(alpha, gd, fk, fNew, criterion, gdNew))
			{
				success = true;
				//std::cout << "Linesearch secceeds: " << alpha << std::endl;
				break;
			}
			else
			{
				alphaMin = kLow * alpha;
				alphaMax = kUp * alpha;

				// Compute the step by using quadratic polynomial interpolation.
				alpha = double(-0.5)*alpha*alpha*gd / (fNew - fk - alpha * gd + EPS);

				// bound checking
				if (alpha < alphaMin || isnan(alpha))
					alpha = alphaMin;
				else if (alpha > alphaMax)
					alpha = alphaMax;
			}
			

		}
		//if (alpha < 1e-8)alpha = 0.01;
		xNew = xk + alpha * dk;
		gNew = func.grad(xNew);
		y = gNew - gk;
		Q_next = yita * Q + 1;
		C = (yita * Q * C + fNew) / Q_next;

		Q = Q_next;
		lambda_saved = -1 * gk.dot(y) / (alpha * gk.dot(gk));
		std::cout << "C: " << C << std::endl;
		//std::cout << "Linesearch 15 times..." << std::endl;
		success = true;
		//std::cout << "Linesearch ends: " << alpha << std::endl;
		
		return alpha;
	}

}