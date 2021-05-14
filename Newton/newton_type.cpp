#include "newton_type.h"
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
	Newton<Ftype>::Newton(): LineSearch<Ftype>(), xOpt(1), gradNorm(1){
		itrNum = 0;
		fMin = 0;
		dim = 1;
	}

	template <typename Ftype>
	Newton<Ftype>::~Newton() {

	}

	//template<typename Ftype>void Newton<Ftype>::optimize(Ftype & func, VectorXd & x0, double tol, int maxItr){  return; }

	template<typename Ftype>
	inline VectorXd Newton<Ftype>::getOptValue() const
	{
		return xOpt;
	}

	template<typename Ftype>
	inline VectorXd Newton<Ftype>::getGradNorm() const
	{
		return gradNorm;
	}

	template<typename Ftype>
	inline double Newton<Ftype>::getFuncMin() const
	{
		return fMin;
	}

	template<typename Ftype>
	int Newton<Ftype>::getItrNum() const
	{
		return itrNum;
	}

	/* The following lines are for damping newton method */

	template <typename Ftype>
	DampingNewton<Ftype>::DampingNewton() : Newton<Ftype>() {
		
	}

	template <typename Ftype>
	DampingNewton<Ftype>::~DampingNewton() {

	}

	template <typename Ftype>
	void DampingNewton<Ftype>::optimize(Ftype & func, VectorXd &x0, double tol, int maxItr)
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
		MatrixXd G(N,N);

		this->setSigma(0.5);
		this->setRho(0.4);

		VectorXd x(x0);
		double fx = func(x);
		double fPrev = fx + 10;
		this->funcNum++;
		
		VectorXd gnorm(maxItr);
		VectorXd g = func.grad(x);
		gnorm[k++] = g.norm();
		std::cout << "Debug point: vnorm" << std::endl;

		//std::cout << "x=" << x << std::endl;

		while ( (gnorm(k-1) > tol * ( 1 + fabs(fx)) ) && (k < maxItr))
		{
			// descent direction
			G = func.hessian(x);
			// Jacobi has been computed within last step
			g = func.grad(x, 0, 0);

			d = G.colPivHouseholderQr().solve(-1 * g);
			// one dimension searching
			alpha = this->getStep(func, x, d, 15, CRITERION_ARMIJO);
			//if (alpha == 0)break;
			
			x = x + alpha * d;
			//std::cout << "alpha=" << alpha << std::endl;
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
		this->itrNum = k-1;
		this->gradNorm.resize(k);
		for (int i = 0; i < k; ++i)
			this->gradNorm[i] = gnorm[i];

		if ((gnorm(k-1) > tol) || fabs(fPrev - fx) > tol)
			this->success = false;

		if (true) {
			std::ofstream outfile;
			outfile.open("results.txt", std::ios::out | std::ios::app);
			std::cout << "the optim is" << this->fMin << std::endl;
			outfile << "Damping Newton" << std::endl;
			outfile << func.getTypeName() << ": \n" << "dimension= " << this->dim << " x*= " << this->xOpt << "\n" << "fmin=" << this->fMin << "itrNum= " << this->itrNum <<" funcNum= " << this->funcNum  << "\n" << std::endl;
		}
	}

	/* The following lines are for Broyden type methods */
	
	template <typename Ftype>
    Broyden<Ftype>::Broyden() : Newton<Ftype>() {

	}

	template <typename Ftype>
	Broyden<Ftype>::~Broyden() {

	}

	template <typename Ftype>
	void Broyden<Ftype>::optimize(Ftype & func, VectorXd &x0,double phi,  double tol, int maxItr)
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

		double alpha, ys, yHy;
		VectorXd d(N), s(N), y(N), v(N), Hy(N), gPrev(N);
		MatrixXd H(N, N);
		H = MatrixXd::Identity(N,N);

		this->setSigma(0.5);
		this->setRho(0.4);

		VectorXd x(x0);
		double fx = func(x);
		double fPrev = fx + 10;
		this->funcNum++;
		VectorXd gnorm(maxItr);
		VectorXd g = func.grad(x);
		gnorm[k++] = g.norm();

		//std::cout << "x=" << x << std::endl;

		while ((gnorm(k - 1) > tol * (1 + fabs(fx))) && (k < maxItr))
		{
			// descent direction


			d = -1 * H * g;
			// one dimension searching
			alpha = this->getStep(func, x, d, 15 , CRITERION_ARMIJO);
			//if (alpha == 0)break;

			s = alpha * d;
			x = x + s;
			//std::cout << "alpha=" << alpha << std::endl;
			//std::cout << "x=" << x << std::endl;
			gPrev = g;
			g = func.grad(x);
			y = g - gPrev;
			
			Hy = H * y;
			ys = y.dot(s);
			yHy = y.dot(Hy);

			if ((ys < EPS) || (yHy < EPS))
				H = MatrixXd::Identity(N, N);
			else
			{
				v = sqrt(yHy) * (s / ys - Hy / yHy);
				if(phi!=0)H = H + s * s.transpose() / ys - Hy * Hy.transpose() / yHy + phi * v * v.transpose();
				else H = H + s * s.transpose() / ys - Hy * Hy.transpose() / yHy;
			}

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

		if ((gnorm(k - 1) > tol) || fabs(fPrev - fx) > tol)
			this->success = false;

		if (true) {
			std::ofstream outfile;
			outfile.open("results.txt", std::ios::out | std::ios::app);
			std::cout << "the optim is" << this->fMin << std::endl;

			outfile << "Broyden, phi= " << phi << std::endl;
			outfile << func.getTypeName() << ": \n" << "dimension= " << this->dim << " x*= " << this->xOpt << "\n" << "fmin=" << this->fMin << "  itrNum= " << this->itrNum  << " funcNum= " << this->funcNum << "\n" << std::endl;
			
			outfile.close();
		}
	}



	template <typename Ftype>
	SR1Newton<Ftype>::SR1Newton() : Newton<Ftype>() {

	}

	template <typename Ftype>
	SR1Newton<Ftype>::~SR1Newton() {

	}

	template <typename Ftype>
	void SR1Newton<Ftype>::optimize(Ftype & func, VectorXd &x0, double tol, int maxItr)
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

		double alpha, e;
		VectorXd d(N), s(N), y(N), Hy(N), t(N), gPrev(N);
		MatrixXd H(N, N);
		H = MatrixXd::Identity(N, N);

		this->setSigma(0.5);
		this->setRho(0.4);

		VectorXd x(x0);
		double fx = func(x);
		double fPrev = fx + 10;
		this->funcNum++;
		VectorXd gnorm(maxItr);
		VectorXd g = func.grad(x);
		gnorm[k++] = g.norm();

		//std::cout << "x=" << x << std::endl;

		while ((gnorm(k - 1) > tol * (1 + fabs(fx))) && (k < maxItr))
		{
			// descent direction


			d = -1 * H * g;
			// one dimension searching
			alpha = this->getStep(func, x, d, 15, CRITERION_ARMIJO);
			//if (alpha == 0)break;

			s = alpha * d;
			x = x + s;
			//std::cout << "alpha=" << alpha << std::endl;
			//std::cout << "x=" << x << std::endl;
			gPrev = g;
			g = func.grad(x);
			y = g - gPrev;

			//Hy = H * y;
			t = s - H * y;

			e = t.dot(y);
			if (e < EPS) H = MatrixXd::Identity(N, N);
			else H = H + t * t.transpose() / (e);

			// check flag for restart
			gnorm[k++] = g.norm();

			fPrev = fx;

			//std::cout << "Debug: function fx=" << fx << std::endl;

			fx = func(x);
			this->funcNum++;
		}

		this->xOpt = x;
		this->fMin = fx;
		this->itrNum = k - 1;
		this->gradNorm.resize(k);
		for (int i = 0; i < k; ++i)
			this->gradNorm[i] = gnorm[i];

		if ((gnorm(k - 1) > tol) || fabs(fPrev - fx) > tol)
			this->success = false;

		if (true) { 
			std::ofstream outfile;
			outfile.open("results.txt", std::ios::out | std::ios::app);
			std::cout << "the optim is" << this->fMin << std::endl;

			outfile << "SR1" << std::endl;
			outfile << func.getTypeName() << ": \n" << "dimension= " << this->dim << " x*= " << this->xOpt << "\n" << "fmin=" << this->fMin <<" itrNum= " << this->itrNum << " funcNum= " << this->funcNum << "\n" << std::endl;
		}

		return;
	}


}
