main:main.cpp  matrice.cpp matrice.hpp dsl.cpp dsl.hpp fonction.cpp fonction.hpp time_manager.cpp time_manager.hpp
	g++ -std=c++17 -O3 -pthread main.cpp fonction.cpp dsl.cpp matrice.cpp time_manager.cpp -o programme
run:
	./programme mnist_test.csv 8 1  
	./programme Microsoft_Stock.csv 8 2 

clean:
	rm programme
