#ifndef _FONCTION_HPP
#define _FONCTION_HPP

#include "dsl.hpp"
#include "time_manager.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>



CNN_Weights_Par_Thread CNNAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, int n_threads);

#endif