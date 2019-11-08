#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void affiche_tab(int * Tab, int len_tab)
{
	for(int i=0; i < len_tab; i++)
	{
		printf("%d\t",Tab[i]);
	}
}


__global__ void merge_Small_k(int* A, int lenA, int* B, int lenB, int* M){

		__shared__ int s_M[1024];
        int i = threadIdx.x; //+ blockIdx.x * 1024;

        int K[2];
        int P[2];
     
        if (i > lenA){

        	K[0] = i - lenA;
        	K[1] = lenA;

        	P[0] = lenA;
        	P[1] = i - lenA; 	
        }
        else{

        	K[0] = 0;
        	K[1] = i;

        	P[0] = i;
        	P[1] = 0; 
        }

        while (true) {
        	int offset = abs(K[1] - P[1])/2;
        	int Q[2];

        	Q[0] = K[0] + offset;
        	Q[1] = K[1] - offset;

        	if(Q[1] >= 0 && Q[0] <= lenB && ( Q[1] == lenA  || Q[0] == 0 || A[Q[1]] <= B[Q[0]] ) ){
        		
        		if(Q[0] == lenB || Q[1] == 0 || A[Q[1]-1] <= B[Q[0]]){
        			
        			if(Q[1] < lenA && ( Q[0] == lenB || A[Q[1]] <= B[Q[0]] ) ){
        				
        				s_M[i] = A[Q[1]];
        			}
        			else{

        				s_M[i] = B[Q[0]];
        			}
        		}
        		else{

        			K[0] = Q[0] + 1;
        			K[1] = Q[1] - 1;
        		}
        	}
        	else{
        		P[0] = Q[0] - 1;
        		P[1] = Q[1] + 1;
        	}

        }

        __syncthreads();
        M[i] = s_M[i];

}

int main(){
	int lenA = 9;
	int lenB = 7;
	int lenM = lenA + lenB;

	// int* A = (int*)malloc(lenA*sizeof(int));
	// int* B = (int*)malloc(lenB*sizeof(int));
	int A[lenA] = {1,2,5,6,6,9,11,15,16};
	int B[lenB] = {4,7,8,10,12,13,14};
	int* M   = (int*)malloc(lenM*sizeof(int));


	int *dev_a, *dev_b, *dev_m;

	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, lenA * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, lenB * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_m, lenM * sizeof(int) ) );


	HANDLE_ERROR( cudaMemcpy( dev_a, A, lenA * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b, B, lenB * sizeof(int), cudaMemcpyHostToDevice ) );


	int blocksize = 1024;

	//float cpu_time;
	float gpu_time;

	Timer timer = Timer();
	timer.start();

	merge_Small_k<<<1,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m);

	timer.add();
	gpu_time = timer.getsum();

	HANDLE_ERROR( cudaMemcpy( M, dev_m, lenM * sizeof(int), cudaMemcpyDeviceToHost ) );

	printf("gpu time: %f\n", gpu_time );
	affiche_tab(M,lenM);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_m);
	free(A);
	free(B);
	free(M);

	return 0;





}


