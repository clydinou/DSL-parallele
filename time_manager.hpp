#ifndef __TIME_MANAGER_HPP
#define __TIME_MANAGER_HPP

// Implémentation portable (Windows/Linux) avec <chrono>
#include <chrono>

// TimePoint : alias pour simplifier l'usage
using TimePoint = std::chrono::steady_clock::time_point;

// Enregistre l'instant courant dans tp
void top_(TimePoint* tp);

// Retourne la durée en secondes entre t1 et t2
double cpu_time_(TimePoint t1, TimePoint t2);

// Retourne un résidu de temps résiduel en microsecondes (compat. ancienne API)
unsigned long get_temp_residuel(void);

#endif
