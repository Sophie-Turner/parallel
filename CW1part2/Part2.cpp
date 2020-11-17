// MatrixMatrix Multiplication
//square matrices only of size NxN
//N=4096
//The arrays are of type float
//Provide the FLOPS value achieved. 

#include <Windows.h> // Needed for SetProcessAffinityMask.
#include <process.h> // Needed for fabs.
#include <chrono> //Needed for timer.
#include <iostream> // Needed for printf. 
#include <immintrin.h> // Needed for AVX.

// Definitions.
#define N 8  // Shortened for testing. Array size = 4096
#define EPSILON 0.01 // Acceptable error.
#define FPOs N*N*N*2 // Number of floating point operations for FLOPS calculation.

// Routine declarations.
void initialise();
void defaultMMM();
void mmm();
unsigned short int compare();
unsigned short int equal(float const a, float const b);
void printMessage(char* s, unsigned short int outcome);

// Variables.
__declspec(align(64)) float A[N][N], B[N][N], C[N][N], test[N][N], Btranspose[N][N]; // Use 64 so we know it will definitely work for all values.
char message[20];


int main() 
{
	double gflops;

	//Pin the current process to the 1st core. Might not need this later...
	BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
	if (success == 0) 
	{
		printf("\nSetProcessAffinityMask failed\n");
		system("pause");
		return -1;
	}

	initialise();

	auto start = std::chrono::high_resolution_clock::now();

	//defaultMMM();
	mmm();

	auto finish = std::chrono::high_resolution_clock::now();

	printMessage(message, compare());

	std::chrono::duration<double> elapsed = finish - start;
	float timeTaken = elapsed.count();
	gflops = ((float)FPOs / timeTaken)/1000000;
	printf("elapsed time = %f seconds \nGFlOPS achieved = %f\n", timeTaken, gflops);

}


void initialise() 
{
// Set up the arrays.
	float p = 0.7264;

	for (unsigned int i = 0; i < N; i++) { 
		for (unsigned int j = 0; j < N; j++) {
			C[i][j] = 0.0;
			test[i][j] = 0.0;
			A[i][j] = (j % 9) + p; 
			B[i][j] = (j % 7) - p; 
		}
	}

}

void defaultMMM() 
{	
// Default MMM to compare with.
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
}

////////////////////////COURSEWORK TASK////////////////////////////

void mmm()
{
// The coursework task.
	__m256 ymm0, ymm1, ymm2; // Wide registers for 8 fp vlaues
	__m128 xmm0; // Wide registers for 4 fp values
	int i, j, k;
	float temp;
	
	for (j = 0; j != N; j++)
		for (k = 0; k != N; k++) {
			Btranspose[k][j] = B[j][k];
		}

	for (i = 0; i != N; i++)
		for (j = 0; j != N; j++) {
			ymm0 = _mm256_setzero_ps();
			for (k = 0; k != ((N / 8) * 8); k += 8) {
				ymm1 = _mm256_load_ps(&A[i][k]); // Load 8 values of the arrays
				ymm2 = _mm256_load_ps(&Btranspose[j][k]);
				ymm0 = _mm256_fmadd_ps(ymm1, ymm2, ymm0); // Reduction (+ and *)
			}

			ymm2 = _mm256_permute2f128_ps(ymm0, ymm0, 1);
			ymm0 = _mm256_add_ps(ymm0, ymm2);
			ymm0 = _mm256_hadd_ps(ymm0, ymm0);
			ymm0 = _mm256_hadd_ps(ymm0, ymm0);
			xmm0 = _mm256_extractf128_ps(ymm0, 0);
			_mm_store_ss(&C[i][j], xmm0);

			for (; k < N; k++) {
				C[i][j] += A[i][k] * Btranspose[j][k];
			}
		}
}

////////////////////////////////////////////////////////////////////////////////

unsigned short int compare() {

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				test[i][j] += A[i][k] * B[k][j];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (equal(C[i][j], test[i][j]) == 1)
				return 1;

	return 0;
}

unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp) < EPSILON)
		return 0; //success
	else
		return 1;
}

void printMessage(char* s, unsigned short int outcome) {

	if (outcome == 0)
		printf("\n\n\r ----- %s output is correct -----\n\r", s);
	else
		printf("\n\n\r -----%s output is INcorrect -----\n\r", s);

}



