#include "Eigen\Dense"
#include "Function.h"
#include "linesearch_more.h"
#include "newton_type.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include "linesearch_more.cpp"
#include "newton_type.cpp"
#include "gradient_type.cpp"
//#include "constant.cpp"


int main(){
	
	clock_t t0, t1, t2, t3, t4, t5;
	std::ofstream timefile;
	timefile.open("time.txt", std::ios::in | std::ios::out | std::ios::app);

	optim::BB<function::Function> BB_Solver;
	optim::CG<function::Function> CG_Solver;

	int n;
	int i, j;
	n = 1000;

	function::Trigonometric Trigo(n);
	std::cout << "Check 1" << std::endl;
	VectorXd x0(n);
	for (i = 0; i < n; i++) {
		x0[i] = 1.0 / n;
	}

	//BB_Solver.optimize(Trigo, x0, 1e-6, 100000);
	//CG_Solver.optimize(Trigo, x0, FR, 10, 1e-6, 100000);

	function::ExtentedPowell EP(n);
	std::cout << "Check 2" << std::endl;
	VectorXd x1(n);
	for (i = 0; i < n / 4; i++) {
		x1[4 * i] = 3;
		x1[4 * i + 1] = -1;
		x1[4 * i + 2] = 0;
		x1[4 * i + 3] = 3;
	}

	//BB_Solver.optimize(EP, x1, 1e-6, 100000);

	function::Tridiagonal Tridiag(n);
	std::cout << "Check 3" << std::endl;
	VectorXd x2(n);
	for (i = 0; i < n; i++) {
		x2[i] = 1;
	}

	//BB_Solver.optimize(Tridiag, x2, 1e-6, 100000);
	//CG_Solver.optimize(Tridiag, x2, FR, 10, 1e-6, 100000);


	//DN_Solver.optimize(CP, x1);
	//BYD_Solver.optimize(CP, x1);
	//BYD_Solver.optimize(CP, x1, 0.5);
	//SR1_Solver.optimize(CP, x1);
	n = 10;
	function::MatrixSquare MS(n);
	std::cout << "Check 4" << std::endl;
	VectorXd x3(n * n);
	for (i = 1; i <= n * n; i++) {
		x3[i-1] = 0.2 * sin(i*i);
	}

	t0 = clock();
	//BB_Solver.optimize(Trigo, x0, 1e-6, 10000);
	t1 = clock();
	timefile << "t1: " << (t1 - t0)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(Trigo, x0, FR, 10, 1e-6, 10000);
	t2 = clock();
	timefile << "t2: " << (t2 - t1)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(Trigo, x0, PRP, 10, 1e-6, 10000);
	t3 = clock();
	timefile << "t3: " << (t3 - t2)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(Trigo, x0, FR_PRP, 10, 1e-6, 10000);
	t4 = clock();
	timefile << "t4: " << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(Trigo, x0, FR_H_PRP, 10, 1e-6, 10000);
	t5 = clock();
	timefile << "t5: " << (t5 - t4)* 1.0 / CLOCKS_PER_SEC << std::endl;
	
	t0 = clock();
	//BB_Solver.optimize(EP, x1, 1e-6, 20000);
	t1 = clock();
	timefile << "t1: " << (t1 - t0)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(EP, x1, FR, 10, 1e-6, 20000);
	t2 = clock();
	timefile << "t2: " << (t2 - t1)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(EP, x1, PRP, 10, 1e-6, 20000, 10);
	t3 = clock();
	timefile << "t3: " << (t3 - t2)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(EP, x1, FR_PRP, 10, 1e-6, 20000, 10);
	t4 = clock();
	timefile << "t4: " << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(EP, x1, FR_H_PRP, 10, 1e-6, 10000);
	t5 = clock();
	timefile << "t5: " << (t5 - t4)* 1.0 / CLOCKS_PER_SEC << std::endl;

	t0 = clock();
	BB_Solver.optimize(Tridiag, x2, 1e-6, 20000);
	t1 = clock();
	timefile << "t1: " << (t1 - t0)* 1.0 / CLOCKS_PER_SEC << std::endl;
	CG_Solver.optimize(Tridiag, x2, FR, 10, 1e-6, 20000);
	t2 = clock();
	timefile << "t2: " << (t2 - t1)* 1.0 / CLOCKS_PER_SEC << std::endl;
	CG_Solver.optimize(Tridiag, x2, PRP, 10, 1e-6, 20000, 10);
	t3 = clock();
	timefile << "t3: " << (t3 - t2)* 1.0 / CLOCKS_PER_SEC << std::endl;
	CG_Solver.optimize(Tridiag, x2, FR_PRP, 10, 1e-6, 20000, 10);
	t4 = clock();
	timefile << "t4: " << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << std::endl;
	CG_Solver.optimize(Tridiag, x2, FR_H_PRP, 10, 1e-6, 10000);
	t5 = clock();
	timefile << "t5: " << (t5 - t4)* 1.0 / CLOCKS_PER_SEC << std::endl;

	t0 = clock();
	//BB_Solver.optimize(MS, x3, 1e-6, 20000);
	t1 = clock();
	timefile << "t1: " << (t1 - t0)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(MS, x3, FR, 10, 1e-6, 20000);
	t2 = clock();
	timefile << "t2: " << (t2 - t1)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(MS, x3, PRP, 10, 1e-6, 20000, 10);
	t3 = clock();
	timefile << "t3: " << (t3 - t2)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(MS, x3, FR_PRP, 10, 1e-6, 20000, 10);
	t4 = clock();
	timefile << "t4: " << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << std::endl;
	//CG_Solver.optimize(MS, x3, FR_H_PRP, 10, 1e-6, 10000);
	t5 = clock();
	timefile << "t5: " << (t5 - t4)* 1.0 / CLOCKS_PER_SEC << std::endl;

	timefile.close();
	
	system("pause");
}