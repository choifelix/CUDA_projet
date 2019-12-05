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
    printf("\n");
}


__global__ void merge_Small_k(int* A, int lenA, int* B, int lenB, int* M){
        printf("entering thread %d\n",threadIdx.x );
        printf("lenA: %d, lenB: %d\n",lenA,lenB );


		__shared__ int s_M[1024];
        int i = threadIdx.x; //+ blockIdx.x * 1024;
        int iter =0;

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

        //bool loop = true;

        printf("thread %d : entering while\n",threadIdx.x);

        if(i < (lenA+lenB) ){
            while (true) {
                iter++;
                
            	int offset = abs(K[1] - P[1])/2;
            	int Q[2];

            	Q[0] = K[0] + offset;
            	Q[1] = K[1] - offset;

                printf("thread %d : iter %d -- %d - %d;%d\n",threadIdx.x, iter,offset,Q[0],Q[1]);

            	if(  (Q[1] >= 0) && (Q[0] <= lenB) && ( (Q[1] == lenA)  || (Q[0] == 0) || (A[Q[1]] > B[Q[0]-1]) ) ){
                    //printf("hello\n");
            		
            		if( (Q[0] == lenB) || Q[1] == 0 || A[Q[1]-1] <= B[Q[0]]){
                        printf("thread %d : iter %d should nreak soon\n",threadIdx.x, iter);
            			
            			if(Q[1] < lenA && ( Q[0] == lenB || A[Q[1]] <= B[Q[0]] ) ){
            				
            				s_M[i] = A[Q[1]];
                            //M[i] = A[Q[1]];
            			}
            			else{

            				s_M[i] = B[Q[0]];
                            //M[i] = B[Q[0]];
            			}
                        //loop = false;
                        break;
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
        }

        //printf("thread %d : done\n",threadIdx.x );

        __syncthreads();
        M[i] = s_M[i];
        //printf("M[%d] = %d\n",i,M[i] );

}


__device__ void pathBig_k(int* A, int lenA, int* startA, int* B, int lenB, int* startB, int* M, int blockId){
    
    __shared__ int s_M[1024];
    int i = threadIdx.x; //+ blockIdx.x * 1024;
    int iter =0;
    

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

    if(i < (lenA+lenB) ){
        while (true) {
            iter++;
            
            int offset = abs(K[1] - P[1])/2;
            int Q[2];

            Q[0] = K[0] + offset;
            Q[1] = K[1] - offset;

            printf("thread %d : iter %d -- %d - %d;%d\n",threadIdx.x, iter,offset,Q[0],Q[1]);

            if(  (Q[1] >= 0) && (Q[0] <= lenB) && ( (Q[1] == lenA)  || (Q[0] == 0) || (A[Q[1]] > B[Q[0]-1]) ) ){
                //printf("hello\n");
                
                if( (Q[0] == lenB) || Q[1] == 0 || A[Q[1]-1] <= B[Q[0]]){
                    printf("thread %d : iter %d should nreak soon\n",threadIdx.x, iter);
                    
                    if(Q[1] < lenA && ( Q[0] == lenB || A[Q[1]] <= B[Q[0]] ) ){
                        
                        s_M[i] = A[Q[1]];
                        //M[i] = A[Q[1]];
                    }
                    else{

                        s_M[i] = B[Q[0]];
                        //M[i] = B[Q[0]];
                    }
                    //loop = false;
                    break;
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

            if( i == lenA+lenB -1 ){
            *startA = Q[1] + 1;
            *startB = Q[0] + 1;

        }
        }

        
    }

    


    __syncthreads();
    M[i + blockId*1024] = s_M[i];
}



__global__ void mergeBig_k(int* A, int lenA, int* B, int lenB, int* M, int nbBlock){
    //int threadId = threadIdx.x;
    //int blockId  = blockIdx.x;
    int startA = 0;
    int startB = 0;
    int mem_startA = startA;
    int mem_startB = startB;
    int local_lenA = lenA;
    int local_lenB = lenB;

    for (int i=0 ; i<nbBlock ; i++){
        printf("------------%d iterration---------",i);
        pathBig_k(A,local_lenA, &startA, B, local_lenB, &startB, M, i);

        //modifie la longueur des listes A et B
        local_lenA -= startA - mem_startA;
        local_lenB -= startB - mem_startB;

        //memorisation deu depart pour l'iteration suivante
        mem_startA = startA;
        mem_startB = startB;
    }

    
}




int main(){
	int lenA = 1024;
	int lenB = 1024;
	int lenM = lenA + lenB;

	// int* A = (int*)malloc(lenA*sizeof(int));
	// int* B = (int*)malloc(lenB*sizeof(int));
	// int A[lenA] = {1,2,5,6,6,9,11,15,16};
	// int B[lenB] = {4,7,8,10,12,13,14};
    int A[lenA];
    int B[lenB];

    for (int i=0 ; i <1024 ;i++){
        A[i] = 2*i;
        B[i] = 2*i +1;
    }
	//int* M;
    //M = (int*)malloc(lenM*sizeof(int));
    int M[lenM]; 


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

    printf("begining sorting\n");
    int nbBlock = lenM/1024 + 1;
	// merge_Small_k<<<1,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m);
    mergeBig_k<<<nbBlock,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m, nbBlock);
    printf("sorted\n");

	timer.add();
	gpu_time = timer.getsum();

    printf("memcopy M\n");
	HANDLE_ERROR( cudaMemcpy( M, dev_m, lenM * sizeof(int), cudaMemcpyDeviceToHost ) );
    printf("memcopy M : done\n");

	printf("gpu time: %f\n", gpu_time );
	affiche_tab(M,lenM);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_m);
	// free(A);
	// free(B);
	// free(M);

	return 0;





}


