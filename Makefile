main:main.cpp  matrice.cpp matrice.h dsl.cpp dsl.h fonctions.c fonctions.h time_manager.c time_manager.h
	g++ -O3 -std=c++11 main.cpp fonction.cpp dsl.cpp matrice.cpp time_manager.cpp -pthread -o main
run:
	./main mnist_test.csv 8

clean:
	rm main
