#pragma once
const double EPS = 1e-14;
const double EPSS = 1e-10;

#define CRITERION_ARMIJO 0
#define CRITERION_GOLDSTEIN 1
#define CRITERION_WOLF 2
#define CRITERION_STRONG_WOLF 3

#define CRITERION_MODIFIED_ARMIJO 4
#define CRITERION_MODIFIED_WOLF 5

bool DELTA(int i, int j);