

#include <cuda_runtime.h>
#include <cutil.h>
#include <stdio.h>
#include <cublas.h>


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/*
void Test()
{
	
	
	const int ROW = 2;
	const int COL = 2;
	float arr[ROW*COL];
	cublasStatus ret;
	int count = 1;
	for(int j=0;j<COL;j++)
	{
		for(int i=0;i<ROW;i++)
		{
			arr[IDX2C(i,j,ROW)] =(float) count++;
			
		}
		
	}

	for(int i=0;i<COL*ROW;i++)
	{
		printf("%f\t", arr[i]);

	}
	printf("\n");
			
	cublasInit();
	float* d_vec;
	float* d_res;
	if(cublasAlloc(ROW*COL, sizeof(float), (void**)&d_vec) != CUBLAS_STATUS_SUCCESS)
	{
		printf("Alloc failed\n");
		cublasShutdown();
		return;
	}
	if(cublasAlloc(ROW*ROW, sizeof(float), (void**)&d_res) != CUBLAS_STATUS_SUCCESS)
	{
		printf("Alloc failed\n");
		cublasShutdown();
		return;
	}
	ret = cublasSetMatrix(ROW, COL, sizeof(float), arr, ROW, d_vec, ROW);

	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf ("data download failed");
		cublasFree (d_vec);
		cublasShutdown();
		return;
	}


	//printf("%d\n", CUBLAS_STATUS_SUCCESS);

//	printf("%d\n", cublasSetVector(10, sizeof(int), arr+40, 1, d_vec+10, 1));
	
	//int buf[100];
	float* buf;
	buf = (float*) malloc(ROW * COL *sizeof(float));
	cublasSgemm('n', 't', ROW, ROW, COL, 1.0f, d_vec, ROW, d_vec, ROW, 0.0f, d_res, ROW);
	printf("Get back...\n");
	ret = cublasGetMatrix(ROW, COL, sizeof(float), d_vec, ROW, buf, ROW);
	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf ("data upload failed");
		cublasFree (d_vec);
		cublasShutdown();
		return;
	}

	float* res;
	res = (float*) malloc(ROW*ROW*sizeof(float));

	ret = cublasGetMatrix(ROW, ROW, sizeof(float), d_res, ROW, res, ROW);
	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf ("result upload failed");
		cublasFree (d_vec);
		cublasShutdown();
		return;
	}
	for(int j=0;j<COL;j++)
	{
		for(int i=0;i<ROW;i++)
		{
			
			printf("%f\t",buf[IDX2C(i,j,ROW)]);
		}
		printf("\n");
	}
	printf("\n");
	for(int j=0;j<ROW;j++)
	{
		for(int i=0;i<ROW;i++)
		{

			printf("%f\t",res[IDX2C(i,j,ROW)]);
		}
		printf("\n");
	}
	printf("\n");

	for(int i=0;i<ROW*ROW;i++)
	{
		printf("%f\t",res[i]);
	}
	cublasFree(d_vec);
	cublasShutdown();
	free(buf);
	free(res);
}
*/

void Test()
{
	cublasInit();

	const int Ah = 3;

	const int Aw = 2;
	const int Bh = Aw;
	const int Bw = 2;

	
	const int Cw = Bw;
	const int Ch = Ah;

	const int sizeA = Ah * Aw;
	const int sizeB = Bh * Bw;
	const int sizeC = Ch * Cw;
	
	float* A;
	float* B;
	float* C;

	CUDA_SAFE_CALL( cudaMallocHost((void**) &A, sizeof(float)*sizeA));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &B, sizeof(float)*sizeB));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &C, sizeof(float)*sizeC));

	int count = 1;
	for(int i = 0;i< Ah;i++)
	{
		for(int j = 0;j < Aw;j++)
		{
			A[IDX2C(i,j,Ah)] = (float) count++;
			printf("%d\t", j*Ah+i);

		}
		
		
	}
	printf("\n");
	count = 1;
	for(int i = 0;i< Bh;i++)
	{
		for(int j=0;j<Bw;j++)
		{
			B[j*Bh+i] = (float) count++;
		}
	}

	for(int i = 0;i< Ah;i++)
	{
		for(int j = 0;j< Aw; j++)
		{
			printf("%f\t", A[i*Aw +j]);
		}
		printf("\n");
	}
	printf("\n");
	for(int i = 0;i< Bh;i++)
	{
		for(int j = 0;j< Bw; j++)
		{
			printf("%f\t", B[i*Bw +j]);
		}
		printf("\n");
	}
	printf("\n");
	
	float* d_A;
	float* d_B;
	float* d_C;
	cublasStatus ret;
	ret = cublasAlloc(sizeA, sizeof(float), (void**)&d_A);
	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("Alloc failed\n");
		cublasShutdown();
		return;
	}
	ret = cublasAlloc(sizeB, sizeof(float), (void**)&d_B);
	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("Alloc failed\n");
		cublasShutdown();
		return;
	}
	ret = cublasAlloc(sizeC, sizeof(float), (void**)&d_C);
	if(ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("Alloc failed\n");
		cublasShutdown();
		return;
	}

	//CUDA_SAFE_CALL(cudaMemcpy(d_A, A, sizeof(float)*sizeA,cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_B, B, sizeof(float)*sizeB,cudaMemcpyHostToDevice));
	cublasSetMatrix(Ah, Aw, sizeof(float), A, Ah, d_A, Ah);
	cublasSetMatrix(Bh, Bw, sizeof(float), B, Bh, d_B, Bh);
	char x[4] = {'n','n','t','t'};
	char y[4] = {'n','t','n','t'};

	for(int i = 0;i<4;i++)
	{
		cublasSgemm(x[i],y[i],Ah, Bw, Aw, 1.0f,  d_A, Ah, d_B, Bh, 0.0f, d_C, Ch );
		cudaThreadSynchronize();
		cublasGetMatrix(Ch, Cw, sizeof(float), d_C, Ch, C, Ch);

		for(int i = 0;i< 3;i++)
		{
			for(int j = 0;j< 3; j++)
			{
				printf("%f\t", C[IDX2C(i,j,Ch)]);
			}
			printf("\n");
		}
		
		printf("%d\n", cublasGetError());
		printf("\n");
	}
	
	//cublasSgemm('n','t',Ah, Bw, Aw, 1.0f,  d_A, Ah, d_B, Bh, 0.0f, d_C, Ch );
	
	
	

	
	

	cublasShutdown();
}



/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include "cublas.h"

/* Matrix size */
#define N  (3)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
						 float beta, float *C)
{
	int i;
	int j;
	int k;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j) {
			float prod = 0;
			for (k = 0; k < n; ++k) {
				prod += A[k * n + i] * B[j * n + k];
			}
			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
}

/* Main */
int Test2()
{    
	cublasStatus status;
	float* h_A;
	float* h_B;
	float* h_C;
	float* h_C_ref;
	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;
	float alpha = 1.0f;
	float beta = 0.0f;

	const int ROW = N;
	const int COL = N-1;
	int n2 = ROW*COL;
	
	int i;
	float error_norm;
	float ref_norm;
	float diff;

	/* Initialize CUBLAS */
	printf("simpleCUBLAS test running..\n");

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	/* Allocate host memory for the matrices */
	h_A = (float*)malloc(n2 * sizeof(h_A[0]));
	if (h_A == 0) {
		fprintf (stderr, "!!!! host memory allocation error (A)\n");
		return EXIT_FAILURE;
	}
	h_B = (float*)malloc(n2 * sizeof(h_B[0]));
	if (h_B == 0) {
		fprintf (stderr, "!!!! host memory allocation error (B)\n");
		return EXIT_FAILURE;
	}
	h_C = (float*)malloc(ROW*ROW * sizeof(h_C[0]));
	if (h_C == 0) {
		fprintf (stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	/* Fill the matrices with test data */
	int count = 1;
	for (i = 0; i < n2; i++) {
		h_A[i] = count;
		h_B[i] = count;
		h_C[i] = count;
		count++;
	}

	for(int i=0;i<ROW;i++)
	{
		for(int j=0;j<COL;j++)
		{
			printf("%f\t", h_A[i*ROW+j]);
		}
		printf("\n");
	}

	/* Allocate device memory for the matrices */
	status = cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device memory allocation error (A)\n");
		return EXIT_FAILURE;
	}
	status = cublasAlloc(n2, sizeof(d_B[0]), (void**)&d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device memory allocation error (B)\n");
		return EXIT_FAILURE;
	}
	status = cublasAlloc(ROW*ROW, sizeof(d_C[0]), (void**)&d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device memory allocation error (C)\n");
		return EXIT_FAILURE;
	}

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device access error (write A)\n");
		return EXIT_FAILURE;
	}
	status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device access error (write B)\n");
		return EXIT_FAILURE;
	}
	status = cublasSetVector(ROW*ROW, sizeof(h_C[0]), h_C, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device access error (write C)\n");
		return EXIT_FAILURE;
	}

	/* Performs operation using plain C code */
	/*
	simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
	h_C_ref = h_C;
	*/
	/* Clear last error */
	cublasGetError();

	/* Performs operation using cublas */
	cublasSgemm('n', 'n', ROW, ROW, COL, alpha, d_A, ROW, d_B, ROW, beta, d_C, ROW);
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return EXIT_FAILURE;
	}

	/* Allocate host memory for reading back the result from device memory */
	/*
	h_C = (float*)malloc(n2 * sizeof(h_C[0]));
	if (h_C == 0) {
		fprintf (stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}
	*/

	/* Read the result back */
	status = cublasGetVector(ROW*ROW, sizeof(h_C[0]), d_C, 1, h_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! device access error (read C)\n");
		return EXIT_FAILURE;
	}
	
	for(int i=0;i<ROW;i++)
	{
		for(int j=0;j<COL;j++)
		{
			printf("%f\t", h_C[i*ROW+j]);
		}
		printf("\n");
	}

	/* Check result against reference */
	/*
	error_norm = 0;
	ref_norm = 0;
	for (i = 0; i < n2; ++i) {
		diff = h_C_ref[i] - h_C[i];
		error_norm += diff * diff;
		ref_norm += h_C_ref[i] * h_C_ref[i];
	}
	error_norm = (float)sqrt((double)error_norm);
	ref_norm = (float)sqrt((double)ref_norm);
	if (fabs(ref_norm) < 1e-7) {
		fprintf (stderr, "!!!! reference norm is 0\n");
		return EXIT_FAILURE;
	}
	printf( "Test %s\n", (error_norm / ref_norm < 1e-6f) ? "PASSED" : "FAILED");
	*/
	/* Memory clean up */
	free(h_A);
	free(h_B);
	free(h_C);
//	free(h_C_ref);
	status = cublasFree(d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! memory free error (A)\n");
		return EXIT_FAILURE;
	}
	status = cublasFree(d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! memory free error (B)\n");
		return EXIT_FAILURE;
	}
	status = cublasFree(d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! memory free error (C)\n");
		return EXIT_FAILURE;
	}

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}
