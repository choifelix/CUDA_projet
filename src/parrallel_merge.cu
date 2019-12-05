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
__device__ void mergeBig_k(int *A, int startA,int lenA, int * B, int startB, int lenB, int * M, int startM){
    __shared__ int s_M[1024];
    int i = threadIdx.x; //+ blockIdx.x * 1024;
    int blockId = blockIdx.x;
    int iter =0;
    int a_top,b_top,a_bottom,index,a_i,b_i;
    index = i;// + startM;
    lenA -= startA;
    lenB -= startB;
    A = &A[startA];
    B = &B[startB];

    if (index > lenA ){

        b_top = index - lenA; //k[0]
        a_top = lenA;     //k[1]  
    }
    else{
        b_top =  0;        //k[0]
        a_top = index;    //k[1]
    }

    a_bottom = b_top;      //P[1]

    // if( i == 5)
    //     printf("block:%d thread:%d : a_top:%d a_bottom:%d b_top:%d\n",blockId,i,a_top,a_bottom,b_top);



    // printf("thread %d : entering while\n",threadIdx.x);

    if( i < (lenA+lenB) ){
         // if( i== 5 )
         //    printf("blockId:%d thread:%d - entering while\n",blockId,i);
        while (true) {
            iter++;

            int offset = abs(a_top - a_bottom)/2;

            a_i = a_top - offset;     //Q[0] = K[0] + offset;
            b_i = b_top + offset;     //Q[1] = K[1] - offset;
            // if(i == 5)
            //     printf("thread %d : iter %d -- %d - %d;%d\n",threadIdx.x, iter,offset,a_i,b_i);

            if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){
                // if(i == 5)
                //     printf("hello\n");

                if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){
                    // if(i == 5)
                    //     printf("thread %d : iter %d should nreak soon\n",threadIdx.x, iter);

                    if(a_i < lenA && ( b_i == lenB || A[a_i] <= B[b_i] ) ){

                        s_M[i] = A[a_i];
                        //M[i] = A[b_i];
                    }
                    else{

                        s_M[i] = B[b_i];
                        //M[i] = B[a_i];
                    }
                    //loop = false;
                    break;
                }
                else{
                    a_top = a_i - 1;     // K[1] = b_i - 1;
                    b_top = b_i + 1;     // K[0] = a_i + 1;
                }
            }
            else{
                a_bottom = a_i +1;      //P[1] = b_i + 1;
            }
        }
        // if( i == 5 )
        // printf("thread:%d - getting out of while\n",i);
    

    //printf("thread %d : done\n",threadIdx.x );

    __syncthreads();
    M[startM + i] = s_M[i];
    //printf("M[%d] = %d\n",i,M[i] );
    

    }
    // if( i == 5)
    //     printf("blockId:%d thread:%d - finish\n",blockId,i);
}


__global__ void pathBig_k(int* A, int lenA, int* B, int lenB, int* M, int nbblock){
    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int i        = blockIdx.x;
    int a_top,b_top,a_bottom,index,a_i,b_i;
  
    int A_start[1024]; // startA pour chaque block
    int B_start[1024]; // startB pour chaque block
  
    A_start[blockId] = lenA;
    B_start[blockId] = lenB;
        
    //for(int i=0 ; i<nbblock ;i++){
    if( threadId == 0 and blockId==0)
        printf("block number %d \n",i);
    index = i * 1024; //indice de l'ement de M par rapport au nlock (initialisation)

    if (index > lenA){

        b_top = index - lenA; //k[0]
        a_top = lenA;     //k[1]  
    }
    else{

        b_top = 0;        //k[0]
        a_top = index;    //k[1]
    }

    a_bottom = b_top;      //P[1]

    //binary search on diag intersections
    // if( threadId == 0 )
    //     printf("block:%d - entering while\n",i);
    while(true){

        int offset = abs(a_top - a_bottom)/2;

        a_i = a_top - offset;     //Q[0] = K[0] + offset;
        b_i = b_top + offset;     //Q[1] = K[1] - offset;

         if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){
            //printf("hello\n");

            if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){
                A_start[i] = a_i;
                B_start[i] = b_i;
                break;
            }
            else{
                a_top = a_i - 1;     // K[1] = b_i - 1;
                b_top = b_i + 1;     // K[0] = a_i + 1;
            }
        }
        else{
            a_bottom = a_i +1;      //P[1] = b_i + 1;
        }
    }
    // if( threadId == 0 )
    //     printf("block:%d - getting out of while\n",i);


    __syncthreads();
    if( threadId == 0 )
        printf("block:%d - evverything is fine till now - %d,%d\n",i,A_start[i],B_start[i]);
    

    mergeBig_k(A,A_start[i],lenA,B,B_start[i],lenB,M,i*1024);
    
}







int main(){
	int lenA = 10000;
	int lenB = 11245;
	int lenM = lenA + lenB;

	// int* A = (int*)malloc(lenA*sizeof(int));
	// int* B = (int*)malloc(lenB*sizeof(int));
	// int A[lenA] = {1,2,5,6,6,9,11,15,16};
	// int B[lenB] = {4,7,8,10,12,13,14};
    int A[lenA];
    int B[lenB];
    srand (time (NULL));

    for (int i=0 ; i <10000 ;i++){
        A[i] = 2*i;
    }

    for (int i=0 ; i <11245 ;i++){
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
    pathBig_k<<<nbBlock,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m, nbBlock);
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


