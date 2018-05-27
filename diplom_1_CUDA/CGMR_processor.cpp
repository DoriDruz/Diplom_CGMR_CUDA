// Итерац. МСГ с регуляризацией на процессоре[, многоядерном процессоре и видеокарте]
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>

using namespace std;

//global
const int S = 134862;

void add_in_file(double nev) {
	ofstream nevyazka;
	nevyazka.open("nevyazka.dat", ios_base::app);
	nevyazka << nev << endl;
	nevyazka.close();
}

void write_in_file(double *vec, double size, string name) {
	clock_t begin_write = clock();

	cout << "Start write in file" << endl;
	ofstream result_file;
	result_file.open(name, ofstream :: trunc); //x_res_from_main.txt

	for (int i = 0; i < size; ++i) {
		result_file << setprecision(16) << vec[i] << endl;
	}

	result_file.close();

	clock_t end_write = clock();
	double write_time = double(end_write - begin_write) / CLOCKS_PER_SEC;
	cout << "Stop write in file: " << write_time << endl;
}

void create_matr(ifstream& file, double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		file >> setprecision(16) >> matr[i];
	}
}

void show_vec(double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		cout << setprecision(16) << matr[i] << "; ";
	}
	cout << endl;
}

void mult_vec_on_num(double *vec, double num, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec[i] * num;
	}
}

//function changing original matrix A
void mult_A_on_alpha_E(double *matr, double alpha) {
	for (int i = 0; i < S; ++i) {
		if (i == 0) {
			matr[i] = matr[i] + alpha;
		}
		else if (i > 0 && i < 247) {
			matr[i * 7 + 1] = matr[i * 7 + 1] + alpha;
		}
		else if (i > 246 && i < 494) {
			matr[i * 7 + 2] = matr[i * 7 + 2] + alpha;
		}
		else if (i > 493 && i < 134368) {
			matr[i * 7 + 3] = matr[i * 7 + 3] + alpha;
		}
		else if (i > 134367 && i < 134615) {
			matr[i * 7 + 4] = matr[i * 7 + 4] + alpha;
		}
		else if (i > 134614 && i < 134861) {
			matr[i * 7 + 5] = matr[i * 7 + 5] + alpha;
		}
		else if (i == 134861) { //944034
			matr[i * 7 + 6] = matr[i * 7 + 6] + alpha;
		}
	}
}

void sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] + vec_two[i];
	}
}

void dif_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] - vec_two[i];
	}
}

void copy_matr(double *m_one, double *m_two) {
	for (int i = 0; i < S * 7; ++i) {
		m_one[i] = m_two[i];
	}
}

void matr_on_vec(double *matr, double *vec, double *res_vec) {
	int second_count = 0;
	int third_count = 0;
	int fourth_count = 0;
	int fifth_count = 0;
	int sixth_count = 0;

	//debug
	//ofstream debug_f;
	//debug_f.open("debug.dat", ios_base::app);

	for (int i = 0; i < S; ++i) {
		if (i == 0) {
			res_vec[i] = \
				matr[0] * vec[0] \
				+ matr[1] * vec[1] \
				+ matr[2] * vec[247] \
				+ matr[3] * vec[494];

			/*debug
			debug_f << i << ": " << res_vec[i] << " = " \
				<< matr[0] << " * " << vec[0] << " + " \
				<< matr[1] << " * " << vec[1] << " + " \
				<< matr[2] << " * " << vec[247] << " + " \
				<< matr[3] << " * " << vec[494] \
				<< endl;
				*/
		}
		else if (i > 0 && i < 247) {
			res_vec[i] = \
				matr[7 * i] * vec[second_count] \
				+ matr[7 * i + 1] * vec[second_count + 1] \
				+ matr[7 * i + 2] * vec[second_count + 2] \
				+ matr[7 * i + 3] * vec[second_count + 248] \
				+ matr[7 * i + 4] * vec[second_count + 495];

			/*debug
			if(i == 1) {
			debug_f << i << ": " << res_vec[i] << " = " \
				<< matr[7 * i] << " * " << vec[second_count] << " + " \
				<< matr[7 * i + 1] << " * " << vec[second_count + 1] << " + " \
				<< matr[7 * i + 2] << " * " << vec[second_count + 2] << " + " \
				<< matr[7 * i + 3] << " * " << vec[second_count + 248] << " + " \
				<< matr[7 * i + 4] << " * " << vec[second_count + 495] \
				<< endl;
			}*/

			second_count++;
		}
		else if (i > 246 && i < 494) {
			res_vec[i] = \
				matr[7 * i] * vec[third_count] \
				+ matr[7 * i + 1] * vec[third_count + 246] \
				+ matr[7 * i + 2] * vec[third_count + 247] \
				+ matr[7 * i + 3] * vec[third_count + 248] \
				+ matr[7 * i + 4] * vec[third_count + 494] \
				+ matr[7 * i + 5] * vec[third_count + 741];

			/*debug
			if (i == 247) {
				debug_f << i << ": " << res_vec[i] << " = " \
					<< matr[7 * i] << " * " << vec[third_count] << " + " \
					<< matr[7 * i + 1] << " * " << vec[third_count + 246] << " + " \
					<< matr[7 * i + 2] << " * " << vec[third_count + 247] << " + " \
					<< matr[7 * i + 3] << " * " << vec[third_count + 248] << " + " \
					<< matr[7 * i + 4] << " * " << vec[third_count + 494] << " + " \
					<< matr[7 * i + 5] << " * " << vec[third_count + 741] \
					<< endl;
			}*/

			third_count++;
		}
		else if (i > 493 && i < 134368) {
			res_vec[i] = \
				matr[7 * i] * vec[fourth_count] \
				+ matr[7 * i + 1] * vec[fourth_count + 247] \
				+ matr[7 * i + 2] * vec[fourth_count + 493] \
				+ matr[7 * i + 3] * vec[fourth_count + 494] \
				+ matr[7 * i + 4] * vec[fourth_count + 495] \
				+ matr[7 * i + 5] * vec[fourth_count + 741] \
				+ matr[7 * i + 6] * vec[fourth_count + 988];

			/*debug
			if (i == 494) {
				debug_f << i << ": " << res_vec[i] << " = " \
					<< matr[7 * i] << " * " << vec[fourth_count] << " + " \
					<< matr[7 * i + 1] << " * " << vec[fourth_count + 247] << " + " \
					<< matr[7 * i + 2] << " * " << vec[fourth_count + 493] << " + " \
					<< matr[7 * i + 3] << " * " << vec[fourth_count + 494] << " + " \
					<< matr[7 * i + 4] << " * " << vec[fourth_count + 495] << " + " \
					<< matr[7 * i + 5] << " * " << vec[fourth_count + 741] << " + " \
					<< matr[7 * i + 6] << " * " << vec[fourth_count + 988] \
					<< endl;
			}*/

			fourth_count++;
		}
		else if (i > 134367 && i < 134615) {
			res_vec[i] = \
				matr[7 * i + 1] * vec[fifth_count + 133874] \
				+ matr[7 * i + 2] * vec[fifth_count + 134121] \
				+ matr[7 * i + 3] * vec[fifth_count + 134367] \
				+ matr[7 * i + 4] * vec[fifth_count + 134368] \
				+ matr[7 * i + 5] * vec[fifth_count + 134369] \
				+ matr[7 * i + 6] * vec[fifth_count + 134615];

			/*debug
			if (i == 134614) {
				cout << i << ", " << fifth_count << ": " << res_vec[i] << " = " \
					<< matr[7 * i + 1] << " * " << vec[fifth_count + 133874] << " + " \
					<< matr[7 * i + 2] << " * " << vec[fifth_count + 134121] << " + " \
					<< matr[7 * i + 3] << " * " << vec[fifth_count + 134367] << " + " \
					<< matr[7 * i + 4] << " * " << vec[fifth_count + 134368] << " + " \
					<< matr[7 * i + 5] << " * " << vec[fifth_count + 134369] << " + " \
					<< matr[7 * i + 6] << " * " << vec[fifth_count + 134615] \
					<< endl;
			}*/

			fifth_count++;
		}
		else if (i > 134614 && i < 134861) {
			res_vec[i] = \
				matr[7 * i + 2] * vec[sixth_count + 134121] \
				+ matr[7 * i + 3] * vec[sixth_count + 134368] \
				+ matr[7 * i + 4] * vec[sixth_count + 134614] \
				+ matr[7 * i + 5] * vec[sixth_count + 134615] \
				+ matr[7 * i + 6] * vec[sixth_count + 134616];

			/*debug
			if (i == 134615) {
				debug_f << i << ": " << res_vec[i] << " = " \
					<< matr[7 * i + 2] << " * " << vec[sixth_count + 134121] << " + " \
					<< matr[7 * i + 3] << " * " << vec[sixth_count + 134368] << " + " \
					<< matr[7 * i + 4] << " * " << vec[sixth_count + 134614] << " + " \
					<< matr[7 * i + 5] << " * " << vec[sixth_count + 134615] << " + " \
					<< matr[7 * i + 6] << " * " << vec[sixth_count + 134616] \
					<< endl;
			}*/

			sixth_count++;
		}
		else if (i == 134861) { //last_element_position = 944034
			res_vec[i] = \
				matr[7 * i + 3] * vec[134367] \
				+ matr[7 * i + 4] * vec[134614] \
				+ matr[7 * i + 5] * vec[134860] \
				+ matr[7 * i + 6] * vec[134861];

			/*debug
			debug_f << i << ": " << res_vec[i] << " = " \
				<< matr[7 * i + 3] << " * " << vec[134367] << " + " \
				<< matr[7 * i + 4] << " * " << vec[134614] << " + " \
				<< matr[7 * i + 5] << " * " << vec[134860] << " + " \
				<< matr[7 * i + 6] << " * " << vec[134861] \
				<< endl;
		}*/

		//debug
		//cout << "i: " << i << " j: " << j  << " res_vec: " << res_vec[i] << " matr: " << matr[3 * i + j] << " vec: " << vec[j] <<  endl;

		}

		//debug
		//debug_f.close();
	}
}

double vec_on_vec(double *vec_one, double *vec_two) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += vec_one[i] * vec_two[i];
	}
	//debug
	//cout << res << endl;
	//system("pause");
	
	return res;
}

double norm_vec(double *vec) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += pow(vec[i], 2);
	}
	return sqrt(res);
}

void copy_vec(double *matr_one, double *matr_two) {
	for (int i = 0; i < S; ++i) {
		matr_one[i] = matr_two[i];
	}
}

void nullify(double *vec) {
	for (int i = 0; i < S; ++i) {
		vec[i] = 0;
	}
}

double check_stop(double *A, double *F, double *X, double Eps) {
	double stop_up = 0;
	double stop_down = 0;
	double *stop_vec = new double[S];
	double *stop_tmp = new double[S];
	
	matr_on_vec(A, X, stop_tmp);
	dif_vec(stop_tmp, F, stop_vec);
	stop_up = norm_vec(stop_vec);
	stop_down = norm_vec(F);
	
	delete(stop_vec);
	delete(stop_tmp);
	
	return stop_up / stop_down;
}

void debug_matr_on_vec(double *X) {
	cout << "Start Debug" << endl;
	ofstream result_file;
	result_file.open("debug.dat", ofstream::trunc);

	for (int i = 0; i < S; ++i) {
		result_file << setprecision(16) << X[i] << endl;
	}

	result_file.close();
}

void CGMR(double *A, double *F, clock_t begin_algo) {
	const double Eps = 0.001;

	//double *rA = new double[S * 7];
	double *x = new double[S];
	double *r = new double[S];
	double *p = new double[S];

	//original A
	double *or_A = new double[S * 7];

	//coef in cycle
	double ak = 0;
	double bk = 0;
	
	//const coef a
	double alpha = 0;

	//tmp
	double *tmp = new double[S];
	double *rApk = new double[S];
	double *stop = new double[S];
	double *r_k1 = new double[S];
	double *x_k1 = new double[S];
	double *p_k1 = new double[S];
	double stop_norm = 0;
	double stop_norm2 = 0;
	double norm = 0;
	double norm2 = 0;
	double vov = 0;
	double stop_eps = 1;

	int count_tmp = 0;

	//clear file with nevyazka
	ofstream nevyazka_file;
	nevyazka_file.open("nevyazka.dat", ofstream::trunc);
	nevyazka_file.close();

	//clear file with debug
	ofstream debug_f;
	debug_f.open("debug.dat", ofstream::trunc);
	debug_f.close();

	//ЗАНУЛИТЬ rApk|or_A?

	//fill all arrays with zeroes
	clock_t begin_zero_filling = clock();
	for (int i = 0; i < S; ++i) {
		x[i] = r[i] = p[i] = 0;
		stop[i] = tmp[i] = r_k1[i] = x_k1[i] = p_k1[i] = 0;
	}
	clock_t end_zero_filling = clock();

	double z_time = double(end_zero_filling - begin_zero_filling) / CLOCKS_PER_SEC;
	cout << "Zero_filling_preperation: " << z_time << endl;

	clock_t begin_make_prep = clock();
	
	//Подготовка перед алгоритмом

	//copy A in or_A
	copy_matr(or_A, A);

	//making const rA
	//rA = A + alpha*E
	mult_A_on_alpha_E(A, alpha);

	//r = b-rA*x0	|rA*x0 = 0| |r = b|
	copy_vec(r, F);
	//p = r;
	copy_vec(p, r);
	
	clock_t end_make_prep = clock();

	double p_time = double(end_make_prep - begin_make_prep) / CLOCKS_PER_SEC;
	cout << "Preperation rA, r, p: " << p_time << endl << endl;
	
	//WHY NEVYAZKA INCREASING ? - Different data in arrays? Make zero iter out of cycle?

	while( !(stop_eps < Eps) && count_tmp < S) {

		count_tmp++;
		cout << "Iteration: " << count_tmp << " / " << S << endl;

		//cout << "Start of iter" << endl;
		clock_t begin_CGMR = clock();

		//ak = norm(r_k)^2 / (rA*p_k, p_k)
		//norm(r_k)^2
			//norm = pow(norm_vec(r), 2);
		norm = vec_on_vec(r, r);
		//rA*p_k
		matr_on_vec(A, p, rApk);
		//debug_matr_on_vec(rApk);
		//write_in_file(F, S, "F_exp.dat");
		//(rA*p_k, p_k)
		vov = vec_on_vec(rApk, p);
		//ak = ...
		ak = (isnan(norm / vov) ? 0 : (norm / vov));
		
		//r_k+1 = r_k - ak*rA*p_k
		//rA*p_k
		//matr_on_vec(A, p, tmp);
			//cout << "rA*pk: " << endl;
			//show_matr(tmp, 3);
		//ak*(rA*p_k)
		mult_vec_on_num(rApk, ak, tmp);
			//cout << "ak*rA*pk: " << endl;
			//show_matr(tmp, 3);
		//r_k+1 = r_k - ...
		dif_vec(r, tmp, r_k1);

		//cout << "r_k: " << endl;
		//show_matr(r, 3);
		//cout << "r_k+1:" << endl;
		//show_matr(r_k1, 3);

		nullify(tmp);
		//x_k+1 = x_k + ak*p_k
			//tmp = p;
			//copy_vec(tmp, p);
		//ak*p_k
		mult_vec_on_num(p, ak, tmp);
		//x_k+1 = x_k + ...
		sum_vec(x, tmp, x_k1);

		//bk = norm(r_k+1)^2 / norm(r_k)^2
		//norm(r_k+1)^2
			//norm = pow(norm_vec(r_k1), 2);
		norm = vec_on_vec(r_k1, r_k1);
		//norm(r_k)^2
			//norm2 = pow(norm_vec(r), 2);
		norm2 = vec_on_vec(r, r);
		//bk = ...
		bk = (isnan(norm / norm2) ? 0 : (norm / norm2));

		nullify(tmp);
		//p_k+1 = r_k+1 + bk*p_k
		//bk*p_k
			//tmp = p;
			//copy_vec(tmp, p);
		mult_vec_on_num(p, bk, tmp);
		//p_k+1 = r_k+1 + ...
		sum_vec(r_k1, tmp, p_k1);

		//show results
		//cout << "alpha: " << alpha << "; ak: " << ak << "; bk: " << bk << endl;

		cout << "X[0]: " << endl;
		show_vec(x_k1, 1);

		//nullify(tmp);
		////stop deal
		////rA*x_k
		//matr_on_vec(or_A, x_k1, stop);
		////rA*x_k - b
		//dif_vec(stop, F, tmp);
		////norm(rA*x_k - b)
		//stop_norm = norm_vec(tmp);
		//	//stop_norm = sqrt(vec_on_vec(tmp, tmp));
		////norm(b) - можно вынести как константу
		//stop_norm2 = norm_vec(F);
		//	//stop_norm2 = sqrt(vec_on_vec(F, F));
		////stop_deal = stop_norm / stop_norm2
		//stop_eps = 0;
		//stop_eps = (isnan(stop_norm / stop_norm2) ? 0 : (stop_norm / stop_norm2));

		stop_eps = check_stop(A, F, x_k1, Eps);

		add_in_file(stop_eps);

		cout << "Nev: " << stop_eps << " " << ((stop_eps < Eps) ? "<" : ">") << " " << Eps << endl << endl;
		//boolalpha << 
		if(stop_eps < Eps || count_tmp > S - 1){
			write_in_file(x_k1, S, "X_1.dat");
		}
		
		//copy for next iter
		//p = p_k1;
		copy_vec(p, p_k1);
		//x = x_k1;
		copy_vec(x, x_k1);
		//r = r_k1;
		copy_vec(r, r_k1);
		//clear vectors
		nullify(stop);
		nullify(p_k1);
		nullify(x_k1);
		nullify(r_k1);
		nullify(tmp);
		nullify(rApk);
		//cout << "End of iter" << endl;
				
		clock_t end_CGMR = clock();

		double CGMR_time = double(end_CGMR - begin_CGMR) / CLOCKS_PER_SEC;
		//cout << "Runtime of iter: " << CGMR_time << endl << endl;

		clock_t end_algo = clock();
		double algo_time = double(end_algo - begin_algo) / CLOCKS_PER_SEC;
		//cout << "Runtime of algo: " << algo_time << endl;
	}
	delete(x);
	delete(r);
	delete(p);
	delete(x_k1);
	delete(r_k1);
	delete(p_k1);
	delete(tmp);
	delete(rApk);
	delete(stop);

}

void main() {
	clock_t begin_algo = clock();

	double *matr_A = new double[S * 7];
	double *matr_F = new double[S];
	
	ifstream A;
	ifstream F;
	
	clock_t begin = clock();

	A.open("A_with_01.dat");				// 134862 * 7	| result.dat
	F.open("F_new.dat");						// 134862		| F.dat

	if (A.is_open() && F.is_open()) {
		create_matr(A, matr_A, S * 7);
		create_matr(F, matr_F, S);
	}
	else {
		cout << "Error opening files" << endl;
	}

	A.close();
	F.close();
	
	CGMR(matr_A, matr_F, begin_algo);
	
	clock_t end = clock();
	double time = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Runtime: " << time << endl;

	delete(matr_A);
	delete(matr_F);

	system("pause");

	//last runtime: 847.723 sec.
	//last runtime: 833.092 sec. / Iteration: 29328
	//last runtime: 792.424 sec. / Iteration: 25619
	//last runtime: 2483.177 sec. / Iteration: 83403 / Eps = 0.001
	//last runtime: 2375.798 sec. / Iteration: 83403 / Eps = 0.001
	//last runtime: 1644.729 sec. / Iteration: 53317 / Eps = 0.01
	//last runtime: 740.28 sec. / Iteration: 25619 / Eps = 0.1
	//last runtime: 254.525 sec. / Iteration: 8627 / Eps = 0.1 / Alpha = 0.5
	//last runtime: ~369 sec. / Iteration: ? / Eps = 0.01 / Alpha = 0.5
	//last runtime: 496.967 sec. / Iteration: 17601 / Eps = 0.001 / Alpha = 0.5
	//last runtime: 564.746 sec. / Iteration: 18385 / Eps = 0.1 / Alpha = 0.05
	//last runtime: 438.88 sec. / Iteration: 15267 / Eps = 0.1 / Alpha = 0.1
	//last runtime: 2375.026 sec. / Iteration: 76618 / Eps = 0.00001 / Alpha = 0.05
	//last runtime: 3773.602 sec. / Iteration: 134862+ / Eps = 0.00001 / Alpha = 0.01
		//with F_new.dat
	//last runtime: 541.725 sec. / Iteration: 19328 / Eps = 0.00001 / Alpha = 0
	//last runtime: 3785.569 sec. / Iteration: 134862+ / Eps = 0.00000001 / Alpha = 0
		//with F_new.dat and A_with_minus_and_01 - не стоит использовать с минусом матрицу А
	//last runtime: 1571.473 sec. / Iteration: 43674 / Eps = 0.001 / Alpha = 0

	//! Посчитать с Eps = 10^-8 но без ограничения по количеству итераций !
}