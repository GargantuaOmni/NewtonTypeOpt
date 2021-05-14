#include "gradient_type.h"
#include "constant.h"
#include "Eigen\Dense"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace Eigen;


namespace optim
{
	template <typename Ftype>
	Gradient<Ftype>::Gradient() : LineSearch<Ftype>(), xOpt(1), gradNorm(1) {
		itrNum = 0;
		fMin = 0;
		dim = 1;
	}

	template <typename Ftype>
	Gradient<Ftype>::~Gradient() {

	}

	//template<typename Ftype>void Newton<Ftype>::optimize(Ftype & func, VectorXd & x0, double tol, int maxItr){  return; }

	template<typename Ftype>
	inline VectorXd Gradient<Ftype>::getOptValue() const
	{
		return xOpt;
	}

	template<typename Ftype>
	inline VectorXd Gradient<Ftype>::getGradNorm() const
	{
		return gradNorm;
	}

	template<typename Ftype>
	inline double Gradient<Ftype>::getFuncMin() const
	{
		return fMin;
	}

	template<typename Ftype>
	int Gradient<Ftype>::getItrNum() const
	{
		return itrNum;
	}

	/* The following lines are for damping newton method */

	template <typename Ftype>
	BB<Ftype>::BB() : Gradient<Ftype>() {

	}

	template <typename Ftype>
	BB<Ftype>::~BB() {

	}


	template <typename Ftype>
	void BB<Ftype>::optimize(Ftype & func, VectorXd &x0, double tol, int maxItr)
	{
		// initialize parameters.
		int k = 0,
			cnt = 0,
			N = x0.rows();

		this->dim = func.GetDim();

		if (N != this->dim) {
			std::cout << "The dimension does not match for initial point." << std::endl;
			return;
		}

		std::cout << "Debug point" << std::endl;

		double alpha;
		VectorXd d(N);

		this->setSigma(0.9);
		this->setRho(0.8);
		this->setYita(0.8);

		VectorXd x(x0);
		double fx = func(x);

		std::cout << "Debug point 2" << std::endl;
		this->setC(fx);

		double fPrev = fx + 10;
		this->funcNum++;

		VectorXd gnorm(maxItr);
		VectorXd g = func.grad(x);
		std::cout << "Debug point: g" << std::endl;
		gnorm[k++] = g.norm();
		std::cout << "Debug point: vnorm" << std::endl;

		//std::cout << "x=" << x << std::endl;

		while ( gnorm(k - 1) > (tol * (1 + fabs(fx))) && (k < maxItr))
		{
			g = func.grad(x);
			d = -g;
			
			// one dimension searching
			alpha = this->getNlsaStep(func, x, d, 20, CRITERION_MODIFIED_ARMIJO);
			//if (alpha == 0)break;

			x = x + alpha * d;
			std::cout << "alpha=" << alpha << std::endl;
			//std::cout << "x="  << x << std::endl;

			// check flag for restart
			gnorm[k++] = g.norm();

			fPrev = fx;

			std::cout << "Debug: function fx=" << fx << std::endl;

			fx = func(x);
			this->funcNum++;
		}

		this->xOpt = x;
		this->fMin = fx;
		this->itrNum = k - 1;
		this->gradNorm.resize(k);
		for (int i = 0; i < k; ++i)
			this->gradNorm[i] = gnorm[i];

		if ((gnorm(k - 1) > tol * (1 + fabs(fx))))
			this->success = false;

		if (true) {
			std::ofstream outfile;
			outfile.open("results.txt", std::ios::out | std::ios::app);
			std::cout << "the optim is" << this->fMin << std::endl;
			outfile << "GBB" << std::endl;
			outfile << "Gradient Residual" << gnorm(k - 1)  << std::endl;
			outfile << "Scaled Gradient Residual" << gnorm(k - 1)/ (1 + fabs(fx)) << std::endl;
			outfile << func.getTypeName() << ": \n" << "dimension= " << this->dim << "\n" << "fmin=" << this->fMin << "itrNum= " << this->itrNum << " funcNum= " << this->funcNum << "\n" << std::endl;
		}
	}


	template <typename Ftype>
	CG<Ftype>::CG() : Gradient<Ftype>() {

	}

	template <typename Ftype>
	CG<Ftype>::~CG() {

	}


	template <typename Ftype>
	void CG<Ftype>::optimize(Ftype & func, VectorXd &x0, int type, int Reinit, double tol, int maxItr, double alpha_0)
	{
		// initialize parameters.
		int k = 0,
			cnt = 0,
			N = x0.rows();

		this->dim = func.GetDim();

		if (N != this->dim) {
			std::cout << "The dimension does not match for initial point." << std::endl;
			return;
		}

		std::cout << "Debug point" << std::endl;

		double alpha;
		double beta;
		VectorXd d(N);
		double u, v, t, w;
		double gg, gs;

		this->setSigma(0.9);
		this->setRho(0.8);
		this->setYita(0.8);

		VectorXd x(x0);
		double fx = func(x);

		std::cout << "Debug point 2" << std::endl;
		this->setC(fx);

		double fPrev = fx + 10;
		this->funcNum++;
		double beta_fr;
		double beta_prp;
		double delta = 1e-6;
		double gamma = 1e-6;

		VectorXd gnorm(maxItr);
		VectorXd g = func.grad(x);
		VectorXd g_ = g;
		d = -g;
		VectorXd g_prev = func.grad(x);
		gnorm[k++] = g.norm();
		std::cout << "Debug point: vnorm" << std::endl;

		//std::cout << "x=" << x << std::endl;

		while (gnorm(k - 1) > 1000 || gnorm(k - 1) > (tol * (1 + fabs(fx))) && (k < maxItr))
		{
			
			
			// one dimension searching
			//if( (type == PRP || type == FR_PRP) && gnorm(k - 1) < 1e-2 )alpha = this->getStep(func, x, d, 20, CRITERION_WOLF, 100 * alpha_0);
			alpha = this->getStep(func, x, d, 20, CRITERION_WOLF, alpha_0);
			//if (alpha == 0)break;

			if (this->success == false)d = -g;

			alpha = this->getStep(func, x, d, 20, CRITERION_ARMIJO, alpha_0);

			x = x + alpha * d;
			std::cout << "alpha=" << alpha << std::endl;
			//std::cout << "x="  << x << std::endl;

			// check flag for restart
			gnorm[k++] = g.norm();

			fPrev = fx;

			g_prev = g;
			g = func.grad(x);
			if (k%Reinit == 0) d = -g;
			else if (type == FR) {
				beta = g.dot(g) / g_prev.dot(g_prev);
				d = -1 * g + beta * d;
			}
			else if (type == PRP) {
				beta = g.dot(g - g_prev) / g_prev.dot(g_prev);
				d = -1 * g + beta * d;
			}
			else if (type == FR_PRP) {
				beta_fr = g.dot(g) / g_prev.dot(g_prev);
				beta_prp = g.dot(g - g_prev) / g_prev.dot(g_prev);
				if (beta_prp < -1 * beta_fr) beta = -1 * beta_fr;
				else if (fabs(beta_prp) < beta_fr) beta = beta_prp;
				else beta = beta_fr;

				d = -1 * g + beta * d;
			}
			else if (type == FR_H_PRP) {
				delta = 4e+10 / sqrt(d.dot(d));
				gamma = 4e+10 / sqrt(g_prev.dot(g_prev));
				g_ = func.grad(x + delta * d) - g;

				t = g_.dot(d) / delta;
				u = g_.dot(g) / delta;
				v = g.dot(func.grad(x + gamma * g) - g) / gamma;

				gg = g.dot(g);
				gs = g.dot(d);

				d = ((u * gs - t * gg) * g + (u * gg - v * gs) * d) / (v * t - u * u);
			}

			std::cout << "Debug: function fx=" << fx << std::endl;

			fx = func(x);
			this->funcNum++;
		}

		this->xOpt = x;
		this->fMin = fx;
		this->itrNum = k - 1;
		this->gradNorm.resize(k);
		for (int i = 0; i < k; ++i)
			this->gradNorm[i] = gnorm[i];

		if ((gnorm(k - 1) > tol * (1 + fabs(fx))))
			this->success = false;

		if (true) {
			std::ofstream outfile;
			outfile.open("results.txt", std::ios::out | std::ios::app);
			std::cout << "the optim is" << this->fMin << std::endl;
			outfile << "Type: " << type << std::endl;
			outfile << "Gradient Residual" << gnorm(k - 1) << std::endl;
			outfile << "Scaled Gradient Residual" << gnorm(k - 1) / (1 + fabs(fx)) << std::endl;
			outfile << func.getTypeName() << ": \n" << "dimension= " << this->dim  << "\n" << "fmin=" << this->fMin << "itrNum= " << this->itrNum << " funcNum= " << this->funcNum << "\n" << std::endl;
		}
	}


}