#pragma once

#include "Eigen\Dense"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string>
#include "constant.h"

#define EVALUATE 1
#define NON_EVALUATE 0

using namespace Eigen;

namespace function 
{
	class Function {
	protected:
		short n;
		VectorXd x;
		double value;
		VectorXd gradient;
		MatrixXd Hessian;
		std::string type;

	public:
		Function(): x(4), gradient(4), Hessian(4, 4)
		{};
		std::string getTypeName() {
			return type;
		}

		virtual const double & operator()(const VectorXd & _x, short Eval = EVALUATE) {
			puts("Wrong virtual function evoked!");
			return value;
			
		}
		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			return gradient;
		}
		virtual const MatrixXd & hessian(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			return Hessian;
		}
		short GetDim() const {
			return n;
		}
	};

	class BrownDennis:public Function{
	protected:
		short m;
		VectorXd f;
		MatrixXd J;
	public:
		BrownDennis(short _m) :f(_m), J(_m,4){
			n = 4;
			m = _m;
			value = (double)0;
			x.resize(4);
			gradient.resize(4);
			Hessian.resize(4, 4);
			type = "BrownDennis";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != 4) { puts("Only 4x1 vector should be accpeted.");  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;

			for (int i = 1; i <= m; i++) {
				double t = (double)i / 5.0;
				f[i - 1] = pow(x[0] + t * x[1] - exp(t), 2) + pow(x[2] + sin(t)*x[3] - cos(t), 2);
				value += f[i - 1] * f[i - 1];
			}
			//puts("Actual evaluation evoked.");

			return;
		}

		virtual void JacobiEvaluate() {
			for (int i = 1; i <= m; i++) {
				double t = (double)i / 5.0;
				J(i - 1, 0) = 2 * (x[0] + t * x[1] - exp(t));
				J(i - 1, 1) = 2 * (x[0] + t * x[1] - exp(t)) * t;
				J(i - 1, 2) = 2 * (x[2] + sin(t) * x[3] - cos(t));
				J(i - 1, 3) = 2 * (x[2] + sin(t) * x[3] - cos(t)) * sin(t);
			}
			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE){
			SetValue(_x);
			
			if(Eval== EVALUATE)Evaluate();

			return value;

		}	

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			gradient = VectorXd::Zero(4);

			for (int i = 1; i <= m; i++) {
				double t = (double)i / 5.0;
				gradient[0] += 2 * J(i - 1, 0) * f(i - 1);
				gradient[1] += 2 * J(i - 1, 1) * f(i - 1);
				gradient[2] += 2 * J(i - 1, 2) * f(i - 1);
				gradient[3] += 2 * J(i - 1, 3) * f(i - 1);
			}

			return gradient;
		}

		virtual const MatrixXd & hessian(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			Hessian = MatrixXd::Zero(4, 4);

			for (int i = 1; i <= m; i++) {
				double t = (double)i / 5.0;
				Hessian(0, 0) += f(i - 1) * 2;
				Hessian(0, 1) += f(i - 1) * 2 * t;
				Hessian(1, 0) += f(i - 1) * 2 * t;
				Hessian(1, 1) += f(i - 1) * 2 * t * t;

				Hessian(2, 2) += f(i - 1) * 2;
				Hessian(2, 3) += f(i - 1) * 2 * sin(t);
				Hessian(3, 2) += f(i - 1) * 2 * sin(t);
				Hessian(3, 3) += f(i - 1) * 2 * sin(t) * sin(t);
			}

			Hessian += J.transpose() * J;
			Hessian = 2 * Hessian;
			return Hessian;
		}

	};


	class DiscreteIntegralEquation :public Function {
	protected:
		short m;
		double h;
		VectorXd f;
		MatrixXd J;
	public:
		DiscreteIntegralEquation(short _m) :f(_m), J(_m, _m) {

			n = _m;
			m = _m;
			h = (double)1.0 / (n + 1);
			value = (double)0;

			x.resize(_m);
			gradient.resize(_m);
			Hessian.resize(_m,_m);
			type = "DiscreteIntegralEquation";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != m) { printf_s("Only %d x 1 vector should be accpeted.\n", m);  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;
			int i, j;

			for (i = 1; i <= m; i++) {
				double ti = (double)i * h;
				double tj;
				double s1 = 0;
				double s2 = 0;
				
				for (j = 1; j <= i; j++) { 
					tj = j * h;
					s1 += tj * pow(x[j - 1] + tj + h, 3);
				}
				for (j = i+1; j <= n; j++) {
					tj = j * h;
					s2 += (1 - tj) * pow(x[j - 1] + tj + h, 3);
				}
				f[i - 1] = x[i - 1] + 0.5 * h * ((1 - ti)*s1 + ti * s2);
				value += f[i - 1] * f[i - 1];
			}
			return;
		}

		virtual void JacobiEvaluate() {
			for (int i = 1; i <= m; i++) {
				double ti = (double)i * h;
				for (int j = 1; j <= n; j++) {
					double tj = (double)j * h;
					J(i - 1, j - 1) = (double)(i == j) + 0.5*h*((j <= i)*(1 - ti)*tj * 3 * pow(x[j - 1] + tj + h, 2) + (j > i)* ti*(1 - tj) * 3 * pow(x[j - 1] + tj + h, 2));
				}
			}
			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			//puts("Jacobi completed.");

			gradient = VectorXd::Zero(n);

			for (int i = 1; i <= m; i++) {
				for (int j = 1; j <= n; j++) {
					gradient[j-1] += 2 * J(i - 1, j - 1) * f(i - 1);
				}
			}

			return gradient;
		}

		virtual const MatrixXd & hessian(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			Hessian = MatrixXd::Zero(n, n);

			//puts("Hessian calculated0.");
			//std::cout << "m=" << m << " n=" << n << std::endl;

			for (int i = 1; i <= m; i++) {
				double ti = (double)i * h;
				for (int j = 1; j <= n; j++) {
					double tj = (double)j * h;
					Hessian(j - 1, j - 1) += f(i - 1)*((i == j) + 3 * h * ( (j<=i)*(1-ti)*tj *(x[j-1] + tj + h) + (j>i)* ti * (1-tj) * (x[j-1] + tj + h) ));
				}			
			}

			//puts("Hessian calculated.");

			Hessian += J.transpose() * J;
			Hessian = 2 * Hessian;
			return Hessian;			
		}

	};


	class CombustionPropane :public Function {
	protected:
		short m;
		VectorXd f;
		MatrixXd J;
	public:
		CombustionPropane() :f(5), J(5, 5) {
			n = 5;
			m = 5;
			value = (double)0;
			x.resize(5);
			gradient.resize(5);
			Hessian.resize(5, 5);
			J = MatrixXd::Zero(5, 5);
			type = "CombustionPropane";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != 5) { puts("Only 4x1 vector should be accpeted.");  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;

			f[0] = x[0]*x[1] + x[0] - 3*x[4];
			f[1] = 2 * x[0] * x[1] + x[0] + 2 * x[1] * x[1] + x[1] * x[2] * x[2] + x[1] * x[2] + x[1] * x[3] + x[1] - x[4];
			f[2] = 2 * x[1] * x[2] * x[2] + x[1] * x[3] + 2 * x[2] * x[2] + x[2] - 8 * x[4];
			f[3] = x[1] * x[3] + 2 * x[3] * x[3] - 4 * x[4];
			f[4] = x[0] * x[1] + x[0] + x[1] * x[1] + x[1] * x[2] * x[2] + x[1] * x[2] + x[1] * x[3] + x[1] + x[2] * x[2] + x[2] + x[3] * x[3] - 1;

			//puts("Actual evaluation evoked.");
			for (int i = 0; i < 5; i++) { value += (f[i] * f[i]);  }


			return;
		}

		virtual void JacobiEvaluate() {
			// Assume that the Jacobi has not been updated in the zero entries!
			J(0, 0) = x[1] + 1;
			J(0, 1) = x[1];
			J(0, 4) = -3;

			J(1, 0) = 2 * x[1] + 1;
			J(1, 1) = 2 * x[0] + 4 * x[1] + x[3] + x[2] * x[2] + x[2] + 1;
			J(1, 2) = 2 * x[1] * x[2] + x[1];
			J(1, 3) = x[1];
			J(1, 4) = -1;

			J(2, 1) = 2 * x[2] * x[2] + x[2];
			J(2, 2) = 4 * x[1] * x[2] + x[1] + 4 * x[2] + 1;
			J(2, 4) = -8;

			J(3, 1) = x[3];
			J(3, 3) = x[1] + 4 * x[3];
			J(3, 4) = -4;

			J(4, 0) = x[1] + 1;
			J(4, 1) = x[0] + 2 * x[1] + x[2] * x[2] + x[2] + x[3] + 1;
			J(4, 2) = 2 * x[1] * x[2] + x[1] + 2 * x[2] + 1;
			J(4, 3) = 2 * x[3] + x[1];

			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			gradient = VectorXd::Zero(5);

			for (int i = 1; i <= 5; i++) {
				gradient[0] += 2 * J(i - 1, 0) * f(i - 1);
				gradient[1] += 2 * J(i - 1, 1) * f(i - 1);
				gradient[2] += 2 * J(i - 1, 2) * f(i - 1);
				gradient[3] += 2 * J(i - 1, 3) * f(i - 1);
				gradient[4] += 2 * J(i - 1, 4) * f(i - 1);
			}

			return gradient;
		}

		virtual const MatrixXd & hessian(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			Hessian = MatrixXd::Zero(5, 5);

			Hessian(0, 1) += f(0);
			Hessian(1, 0) += f(0);

			Hessian(0, 1) += 2 * f(1);
			Hessian(1, 0) += 2 * f(1);
			Hessian(2, 2) += 4 * f(1);
			Hessian(2, 3) += (2 * x[2] + 1) * f(1);
			Hessian(2, 4) += f(1);
			Hessian(3, 2) += (2 * x[2] + 1) * f(1);
			Hessian(4, 2) += f(1);

			Hessian(1, 2) += (4 * x[2] + 1) * f(2);
			Hessian(2, 1) += (4 * x[2] + 1) * f(2);
			Hessian(2, 2) += (4 * x[1] + 4) * f(2);

			Hessian(1, 3) += f(3);
			Hessian(3, 1) += f(3);
			Hessian(3, 3) += 4 * f(3);

			Hessian(0, 1) += f(4);
			Hessian(1, 0) += f(4);
			Hessian(1, 1) += 2 * f(4);
			Hessian(1, 2) += (2 * x[2] + 1) * f(4);
			Hessian(1, 3) += f(4);
			Hessian(2, 1) += (2 * x[2] + 1) * f(4);
			Hessian(2, 2) += (2 * x[1] + 2) * f(4);
			Hessian(3, 3) += 2 * f(4);
			Hessian(3, 1) += f(4);

			Hessian += J.transpose() * J;
			Hessian = 2 * Hessian;
			return Hessian;
		}

	};

	class Trigonometric :public Function {
	protected:
		short m;
		//double h;
		VectorXd f;
		MatrixXd J;
	public:
		Trigonometric(short _m) :f(_m), J(_m, _m) {

			n = _m;
			m = _m;
			//h = (double)1.0 / (n + 1);
			value = (double)0;

			x.resize(_m);
			gradient.resize(_m);
			type = "Trigonometric";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != m) { printf_s("Only %d x 1 vector should be accpeted.\n", m);  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;
			int i, j;

			for (i = 1; i <= n; i++) {
				f[i - 1] = n + i;
				for (j = 1; j <= m; j++) {
					f[i - 1] -= DELTA(i, j) * sin(x[j - 1]) + (DELTA(i, j) * i + 1) * cos(x[j - 1]);
				}
				value += f[i - 1] * f[i - 1];
			}
			return;
		}

		virtual void JacobiEvaluate() {
			for (int i = 1; i <= m; i++) {
				for (int j = 1; j <= n; j++) {
					J(i - 1, j - 1) = (double)(-1 * DELTA(i,j) * cos(x[j-1]) + (i * DELTA(i,j) + 1) * sin(x[j-1]));
				}
			}
			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();
			if (Jacobi == EVALUATE)JacobiEvaluate();

			//puts("Jacobi completed.");

			gradient = VectorXd::Zero(n);

			for (int i = 1; i <= m; i++) {
				for (int j = 1; j <= n; j++) {
					gradient[j - 1] += 2 * J(i - 1, j - 1) * f(i - 1);
				}
			}

			return gradient;
		}

	};


	class ExtentedPowell :public Function {
	protected:
		short m;
	public:
		ExtentedPowell(short _n) {

			n = _n;
			if (n % 4 !=0) { puts("n could not be divided by 4!"); }
			m = n / 4;
			
			value = (double)0;

			x.resize(n);
			gradient.resize(n);
			type = "ExtentedPowell";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != n) { printf_s("Only %d x 1 vector should be accpeted.\n", n);  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;
			int i, j;

			for (j = 1; j <= m; j++) {
				value += pow(x[4 * j - 4] + 10 * x[4 * j - 3], 2) + 5 * pow(x[4 * j - 2] - x[4 * j - 1], 2) + pow(x[4 * j - 3] - 2 * x[4 * j - 2], 4) + 10 * pow(x[4 * j - 4] - x[4 * j - 1], 4);
			}
			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			//puts("Jacobi completed.");

			gradient = VectorXd::Zero(n);

			int i, j;
			for (i = 1; i <= n; i++) {
				switch (i % 4) {
				case 1:gradient[i - 1] = 2 * (x[i - 1] + 10 * x[i]) + 40 * pow(x[i - 1] - x[i + 2], 3); break;
				case 2:gradient[i - 1] = 20 * (x[i - 2] + 10 * x[ i - 1]) + 4 * pow(x[i - 1] - 2 * x[i], 3); break;
				case 3:gradient[i - 1] = 10 * (x[i - 1] - x[i]) - 8 * pow(x[i - 2] - 2 * x[i - 1], 3); break;
				case 0:gradient[i - 1] = -10 * (x[i - 2] - x[i - 1]) - 40 * pow(x[i - 4] - x[i - 1], 3); break;
				}
			}

			return gradient;
		}

	};

	class Tridiagonal :public Function {
	protected:
		short m;
	public:
		Tridiagonal(short _n) {

			n = _n;
			m = n;

			value = (double)0;

			x.resize(n);
			gradient.resize(n);
			type = "Tridiagonal";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != n) { printf_s("Only %d x 1 vector should be accpeted.\n", n);  return; }
			x = _x;
			return;
		}

		virtual void Evaluate() {
			value = 0;
			int i, j;

			for (i = 2; i <= n; i++) {
				value += i * pow(2 * x[i - 1] - x[i - 2], 2);
			}
			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			//puts("Jacobi completed.");

			gradient = VectorXd::Zero(n);

			int i;
			for (i = 2; i <= n-1; i++) {
				gradient[i - 1] = 4 * i * (2 * x[i - 1] - x[i - 2]) - 2 * (i + 1) * (2 * x[i] - x[i-1]);
			}
			gradient[0] = -4 * (2 * x[1] - x[0]);
			gradient[n - 1] = 4 * n * (2 * x[n - 1] - x[n - 2]);

			return gradient;
		}

	};


	class MatrixSquare :public Function {
	protected:
		short m;
		VectorXd f;
		VectorXd a;

	public:
		MatrixSquare(short _m):f(_m*_m), a(_m*_m) {

			MatrixXd A(_m, _m);
			MatrixXd B(_m, _m);

			m = _m;
			n = m*m;

			value = (double)0;

			a = VectorXd::Zero(n);

			int i, j;
			for (i = 0; i < m; i++) {
				for (j = 0; j < m; j++) {
					B(i, j) = sin((i*m + j + 1)*(i*m + j + 1));
				}
			}

			A = B * B;

			for (i = 0; i < m; i++) {
				for (j = 0; j < m; j++) {
					a[i*m + j] = A(i, j);
				}
			}

			x.resize(n);
			gradient.resize(n);
			Hessian.resize(n, n);
			type = "MatrixSquare";
		}

		virtual void SetValue(const VectorXd & _x) {
			//Only accept column vector
			if (_x.cols() != 1 || _x.rows() != n) { printf_s("Only %d x 1 vector should be accpeted.\n", n);  return; }
			x = _x;
			return;
		}

		void SetA(const VectorXd & _x) {
			a = VectorXd::Zero(n);
			int l;
			int k;
			int j;
			for (int i = 0; i < n; i++) {
				l = i % m;
				k = int(i / m);
				for (j = 0; j < m; j++) {
					a[i] += _x[j + l * m] * _x[k + j * m];
				}
			}
		}

		virtual void Evaluate() {
			value = 0;
			int i, j;
			int l;
			int k;

			for (i = 0; i < n; i++) {
				l = i % m;
				k = int(i / m);
				f[i] = a[i];
				for (j = 0; j < m; j++) {
					f[i] -= x[j + l * m] * x[k + j * m];
				}
				value += f[i] * f[i];
			}

			

			return;
		}

		virtual const double & operator() (const VectorXd & _x, short Eval = EVALUATE) {
			SetValue(_x);

			if (Eval == EVALUATE)Evaluate();

			return value;

		}

		virtual const VectorXd & grad(const VectorXd & _x, short Eval = EVALUATE, short Jacobi = EVALUATE) {
			SetValue(_x);

			//puts("Jacobi completed.");

			gradient = VectorXd::Zero(n);

			int j, k, l, s, u;
			double t;
			for (j = 0; j < n; j++) {
				l = j % m;
				k = j / m;
				t = 0;
				for (s = 0; s < m; s++) {
					t = a[k*m + s];
					for (u = 0; u < m; u++) {
						t -= x[k * m + u] * x[s + u * m];
					}
					t *= 2 * (-1 * x[l*m + s]);
					gradient[j] += t;
				}
				for (s = 0; s < m; s++) {
					t = a[s*m + l];
					for (u = 0; u < m; u++) {
						t -= x[s * m + u] * x[l + u * m];
					}
					t *= 2 * (-1 * x[s*m + k]);
					gradient[j] += t;
				}
			}

			return gradient;
		}

	};
}
