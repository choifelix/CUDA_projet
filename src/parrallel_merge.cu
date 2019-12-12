//  created on Thu 7 - 11 - 2019
// 
//              by 
//  Felix CHOI and Assef BOINA
// 
//          Supersived by
//Roman Iakymchuk - Lokmane ABBAS TURKI
//
//          Work based on 
//  GPU Merge Path - A GPU Merging Algorithm 
//              by
//  O.Green, R. McColl and D.A. Bader
// 
//  Sorbonne University - Polytech Sorbonne
// 
// all right reserved
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



int cudaMemoryTest()
{
    const unsigned int N = 2097151;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    HANDLE_ERROR(cudaMalloc((int**)&d_a, bytes));

    memset(h_a, 0, bytes);
    HANDLE_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));

    return 0;
}

void affiche_tab(int * Tab, int len_tab)
{
	for(int i=0; i < len_tab; i++)
	{
		printf("%d\t",Tab[i]);
	}
    printf("\n");
}


void affiche_Batchtab(int * Tab, int N, int d)
{
    for(int i=0; i < N; i++)
    {
        printf("%d {",i);
        for(int j=0 ; j<d ; j++){
            printf("%d\t",Tab[i*d +j]);
        }
        printf("}\n");
    }
    printf("\n");
}



__global__ void merge_Small_k(int* A, int lenA, int* B, int lenB, int* M){

	__shared__ int s_M[1024]; //resultat en local
    int i = threadIdx.x; 
    int iter =0;

    int K[2];
    int P[2];


    //initialisation des encadrements des valeurs possibles
 
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



    if(i < (lenA+lenB) ){ //condition sur les threads entrant dans la boucle
        //recherche sur la diagonale
        while (true) {
            iter++;
            
        	int offset = abs(K[1] - P[1])/2;
        	int Q[2];

        	Q[0] = K[0] + offset;
        	Q[1] = K[1] - offset;


        	if(  (Q[1] >= 0) && (Q[0] <= lenB) && ( (Q[1] == lenA)  || (Q[0] == 0) || (A[Q[1]] > B[Q[0]-1]) ) ){
        		
        		if( (Q[0] == lenB) || Q[1] == 0 || A[Q[1]-1] <= B[Q[0]]){
        			
        			if(Q[1] < lenA && ( Q[0] == lenB || A[Q[1]] <= B[Q[0]] ) ){
        				
        				s_M[i] = A[Q[1]];
        			}
        			else{

        				s_M[i] = B[Q[0]];
        			}
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

    __syncthreads();    //synchronisation des threads
    M[i] = s_M[i];      //ecriture du resultat

}






/*##########################################################################
                        Merge Path on any size
                            using 2 kernels
/###########################################################################*/

__device__ void mergeBig_k(int *A, int startA,int lenA, int * B, int startB, int lenB, int * M, int startM){
   
    __shared__ int s_M[1024];
    int i = threadIdx.x; 
    int blockId = blockIdx.x;
    int iter =0;
    int a_top,b_top,a_bottom,index,a_i,b_i;

    //initialisation des variables pour le tri local
    index = i;
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



    if( i < (lenA+lenB) ){  //les threads non concerne ne travaillent pas -> sinon loop infini
        while (true) {
            iter++;

            int offset = abs(a_top - a_bottom)/2;

            a_i = a_top - offset;     //Q[0] = K[0] + offset;
            b_i = b_top + offset;     //Q[1] = K[1] - offset;

            if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){

                if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){

                    if(a_i < lenA && ( b_i == lenB || A[a_i] <= B[b_i] ) ){

                        s_M[i] = A[a_i];
                    }
                    else{

                        s_M[i] = B[b_i];
                    }
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

    __syncthreads();
    M[startM + i] = s_M[i];   
    }

}


__global__ void pathBig_k(int* A, int lenA, int* B, int lenB, int* M){
    int threadId = threadIdx.x;
    int i        = blockIdx.x;
    int a_top,b_top,a_bottom,index,a_i,b_i;
  
    int A_start; // startA pour chaque block
    int B_start; // startB pour chaque block
  
    A_start = lenA;
    B_start = lenB;


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

    while(true){

        int offset = abs(a_top - a_bottom)/2;

        a_i = a_top - offset;     //Q[0] = K[0] + offset;
        b_i = b_top + offset;     //Q[1] = K[1] - offset;

         if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){
            //printf("hello\n");

            if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){
                A_start = a_i;
                B_start = b_i;
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


    __syncthreads();
    mergeBig_k(A,A_start,lenA,B,B_start,lenB,M,i*1024);
    
}



/*##########################################################################
                        Batch Merge on size d on N Batch

/###########################################################################*/

__global__ void mergeSmallBatch_k(int *list_A, int* list_lenA, int*list_B, int *list_lenB, int*list_M, int d, int N){
    int tidx    = threadIdx.x%d; //thread ne traite pas les elements de M_i d'indice sup a d (n'existe pas)
    int BlockId = blockIdx.x;
    int Qt      = (threadIdx.x - tidx)/d; //numero du batch localement au bloc.
    int gbx     = Qt + blockIdx.x*(blockDim.x/d); // indice i du M_i sur M 


//---------calcul du nombre de partition traite par block----------------
    int nbPartPerBlock;
    if(1024/d < N){
        nbPartPerBlock = 1024/d;
    }
    else{
        nbPartPerBlock = N;
    }
    
//--------------calcul nombre block utilise---------------------------
    int nbblock = (N +nbPartPerBlock-1) / nbPartPerBlock;
    
//-----------calcul du nombre de thread utilise par un block----------
    int threadMax;

    if(BlockId == nbblock -1){
        threadMax = (N - (nbPartPerBlock*BlockId) ) *d;
    }
    else{
        if(1024/d < N){
            threadMax = d * (1024/d);
        }
        else{
            threadMax =  N*d;
        }
    }


//------------reconstruction des listes locales----------------
    int startA = 0;
    int startB = 0;

    for(int j=0 ; j<gbx ; j++){
        startA += list_lenA[j];
        startB += list_lenB[j];
    }
    
    int* A = &list_A[startA];
    int* B = &list_B[startB];
    int lenA = list_lenA[gbx];
    int lenB = list_lenB[gbx];
  
    

//--------------- algorithme de tri -------------------
    __shared__ int s_M[1024];
    int i = threadIdx.x; 
    int iter =0;


    int K[2];
    int P[2];

    if (i > lenA){

        K[0] = tidx - lenA;
        K[1] = lenA;

        P[0] = lenA;
        P[1] = tidx - lenA;    
    }
    else{

        K[0] = 0;
        K[1] = tidx;

        P[0] = tidx;
        P[1] = 0; 
    }


    if( (BlockId < nbblock) and (i < threadMax) ){
 
        while (true) {
            iter++;

            int offset = abs(K[1] - P[1])/2;
            int Q[2];

            Q[0] = K[0] + offset;
            Q[1] = K[1] - offset;

            if(  (Q[1] >= 0) && (Q[0] <= lenB) && ( (Q[1] == lenA)  || (Q[0] == 0) || (A[Q[1]] > B[Q[0]-1]) ) ){

                if( (Q[0] == lenB) || Q[1] == 0 || A[Q[1]-1] <= B[Q[0]]){

                    if(Q[1] < lenA && ( Q[0] == lenB || A[Q[1]] <= B[Q[0]] ) ){
                        s_M[i] = A[Q[1]];
                    }
                    else{
                        s_M[i] = B[Q[0]];
                    }

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
    

    __syncthreads();
    list_M[gbx*d + tidx] = s_M[i];
    }
}






/*##########################################################################
                        Tests functions

/###########################################################################*/

void construct_input(int **A, int*lenA, int ** B, int *lenB, int d , int N)
{
    int i=0; // compteur de 0 à N-1

    srand (time (NULL));
    //tant que i n'est pas égal au nombre de partionnement N
    while(i<N) 
    {
        
        lenA[i]=rand()%(d -1) + 1 ;
        lenB[i]= d-lenA[i];

        A[i] = (int*)malloc( (lenA[i]) *sizeof(int));
        B[i] = (int*)malloc( (lenB[i]) *sizeof(int));

        for (int j=0;j<d;j++)
            {
                if (j< lenA[i])
                    A[i][j] = j*2;
                if (j < lenB[i])
                    B[i][j] = j*2 + 1;
            }
      i++;
    }


}

void affiche_list(int ** T, int * lenT,int N)
{
    int i,j;
    for(i=0;i<N;i++)
    {
        printf("len de ma sous liste [%d] = %d  \n",i,lenT[i]);
        printf("(\t");
        for (j=0;j<lenT[i];j++)
        {
         printf("%d\t",T[i][j]);
        }
        printf(")\n");
    }
    printf("\n");
}


void convert2D_to1D_array(int **A, int*lenA, int N, int *A_1d){

    int index = 0;
    for(int i=0 ; i<N ; i++){
        for(int j=0 ; j<lenA[i] ; j++){
            A_1d[index] = A[i][j];
            index ++;
        }
    }

}


void test_batchMerge_deterministic(int d,int N){  
    printf("----------------------------------------\n");
    printf("------ begining Batch merge sort -------\n");
    printf("------- ( deterministic lists ) --------\n");
    printf("----------------------------------------\n");


//---------initialisation des tableaux-----------------

    int **A;
    int **B;

    int *lenA;
    int *lenB;
    int *M;


    A =(int **)malloc( N *sizeof(int*));
    B =(int **)malloc( N *sizeof(int*));


    lenA =(int *)malloc( N *sizeof(int));
    lenB =(int *)malloc( N *sizeof(int));
    M =(int *)malloc( N*d *sizeof(int));


    int sizeA = 0;
    int sizeB = 0;


//------------remplissage de A et B----------------

    for (int i=0 ;i<N ; i++){
        A[i] = (int*)malloc( (d/2 -1) *sizeof(int));
        B[i] = (int*)malloc( (d/2 +1) *sizeof(int));
        for(int j=0 ; j<d/2 +1 ; j++){
            if(j < d/2 - 1){
                A[i][j] = j*2;
                sizeA++;
            }
            B[i][j] = j*2 + 1;
            
            sizeB++;

        }
        lenA[i] = d/2 - 1;
        lenB[i] = d/2 + 1;
    }
  

//---------conversion des tableaux 2D em tableaux 1D-------

    int * A_1d;
    A_1d = (int*)malloc(sizeA*sizeof(int));
    convert2D_to1D_array(A,lenA,N, A_1d);

    int * B_1d;
    B_1d = (int*)malloc(sizeB*sizeof(int));
    convert2D_to1D_array(B,lenB,N, B_1d);

    printf("CPU variable allocated and initialized\n");
    
  
//--------allocation des tableaux sur le device---------
    int *dev_a, *dev_b, *dev_m;
    int * dev_lenA, *dev_lenB; 

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, sizeA * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, sizeB * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_m,   N*d * sizeof(int) ) );

    HANDLE_ERROR( cudaMemcpy( dev_a, A_1d, sizeA * sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, B_1d, sizeB * sizeof(int), cudaMemcpyHostToDevice ) );

  

    HANDLE_ERROR( cudaMalloc( (void**)&dev_lenA, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_lenB, N * sizeof(int) ) );

    HANDLE_ERROR( cudaMemcpy( dev_lenA, lenA, N * sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_lenB, lenB, N * sizeof(int), cudaMemcpyHostToDevice ) );


    printf("GPU variable allocated and initialized\n");

    
   
//--------calcul du nombre de threads et nombre de blocs------------
    int threadsPerBlock = d * (1024/d);
    int nbBlock = (N*d + threadsPerBlock-1) / threadsPerBlock;

    printf("N=%d , d=%d , threads= %d , blocs= %d\n",N,d,threadsPerBlock,nbBlock);

//------- calculs ---------

    float gpu_time;

    Timer timer = Timer();
    timer.start();

    printf("begining sorting\n");
    
    mergeSmallBatch_k<<<nbBlock,threadsPerBlock>>>(dev_a, dev_lenA, dev_b, dev_lenB, dev_m, d, N);
    cudaDeviceSynchronize();

    timer.add();
    gpu_time = timer.getsum();

    HANDLE_ERROR( cudaMemcpy( M, dev_m, N*d * sizeof(int), cudaMemcpyDeviceToHost ) );
    
    printf("memcopy M : done\n");

    affiche_Batchtab(M,N,d);

    printf("gpu time: %f\n", gpu_time );
    
    
//------- liberation memoire ---------------
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_m);

    cudaFree(dev_lenA);
    cudaFree(dev_lenB);
  
  
    for(int i=0 ; i<N ; i++){
        free(A[i]);
        free(B[i]);
    }

    free(A);
    free(B);

    free(lenA);
    free(lenB);
    free(M);

    free(A_1d);
    free(B_1d);
};


void test_batchMerge_rand(int d,int N){  
    printf("----------------------------------------\n");
    printf("------ begining Batch merge sort -------\n");
    printf("----------- ( random lists ) -----------\n");
    printf("----------------------------------------\n");


//---------initialisation des tableaux-----------------]

    int **A;
    int **B;

    int *lenA;
    int *lenB;
    int *M;


    A =(int **)malloc( N *sizeof(int*));
    B =(int **)malloc( N *sizeof(int*));


    lenA =(int *)malloc( N *sizeof(int));
    lenB =(int *)malloc( N *sizeof(int));
    M =(int *)malloc( N*d *sizeof(int));


//------------remplissage de A et B----------------

    construct_input(A,lenA,B,lenB,d,N);

//---------calcul nombre total d'elements de A et B------------

    int sizeA = 0;
    int sizeB = 0;

    for(int i=0 ; i<N ; i++){
        sizeA += lenA[i];
        sizeB += lenB[i];
    }


//---------conversion des tableaux 2D em tableaux 1D-------

    int * A_1d;
    A_1d = (int*)malloc(sizeA*sizeof(int));
    convert2D_to1D_array(A,lenA,N, A_1d);

    int * B_1d;
    B_1d = (int*)malloc(sizeB*sizeof(int));
    convert2D_to1D_array(B,lenB,N, B_1d);

    printf("CPU variable allocated and initialized\n");
  
  
//--------allocation des tableaux sur le device---------

    int *dev_a, *dev_b, *dev_m;
    int * dev_lenA, *dev_lenB; 

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, sizeA * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, sizeB * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_m,   N*d * sizeof(int) ) );

    HANDLE_ERROR( cudaMemcpy( dev_a, A_1d, sizeA * sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, B_1d, sizeB * sizeof(int), cudaMemcpyHostToDevice ) );

  

    HANDLE_ERROR( cudaMalloc( (void**)&dev_lenA, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_lenB, N * sizeof(int) ) );

    HANDLE_ERROR( cudaMemcpy( dev_lenA, lenA, N * sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_lenB, lenB, N * sizeof(int), cudaMemcpyHostToDevice ) );


    printf("GPU variable allocated and initialized\n");

    
   
//--------calcul du nombre de threads et nombre de blocs------------

    int threadsPerBlock = d * (1024/d);
    int nbBlock = (N*d + threadsPerBlock-1) / threadsPerBlock;

    printf("N=%d , d=%d , threads= %d , blocs= %d\n",N,d,threadsPerBlock,nbBlock);


//------- calculs ---------

    float gpu_time;

    Timer timer = Timer();
    timer.start();

    printf("begining sorting\n");

    
    mergeSmallBatch_k<<<nbBlock,threadsPerBlock>>>(dev_a, dev_lenA, dev_b, dev_lenB, dev_m, d, N);
    cudaDeviceSynchronize();

    timer.add();
    gpu_time = timer.getsum();

 
    HANDLE_ERROR( cudaMemcpy( M, dev_m, N*d * sizeof(int), cudaMemcpyDeviceToHost ) );
    
    printf("memcopy M : done\n");

    // affiche_Batchtab(M,N,d);

    printf("gpu time: %f\n", gpu_time);
    
    
    
    
//------- liberation memoire ---------------

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_m);

    cudaFree(dev_lenA);
    cudaFree(dev_lenB);
  
  
    for(int i=0 ; i<N ; i++){
        free(A[i]);
        free(B[i]);
    }


    free(A);
    free(B);

    free(lenA);
    free(lenB);
    free(M);

    free(A_1d);
    free(B_1d);

};




void test_PathMerge(int d){

    printf("----------------------------------------\n");
    printf("------ begining Path merge sort -------\n");
    printf("----------------------------------------\n");

//---------initialisation des tableaux-----------------]

    int lenM = d;
    int lenA = rand()%(lenM -1) + 1;
    int lenB = lenM - lenA;
    

    int *A;
    int *B;
    int* M;

    A = (int*)malloc(lenA*sizeof(int));
    B = (int*)malloc(lenB*sizeof(int));
    M = (int*)malloc(lenM*sizeof(int));

//------------remplissage de A et B----------------

    // for (int i=0 ; i <lenA ;i++){
    //     A[i] = 2*i;
    // }

    // for (int i=0 ; i <lenB ;i++){
    //     B[i] = 2*i +1;
    // }

    A[1] = rand()%10;
    B[1] = rand()%10;

    for (int i=1 ; i <lenA ;i++){
        A[i] = A[i-1] + rand()%10;
    }

    for (int i=1 ; i <lenB ;i++){
        B[i] = B[i-1] + rand()%10;
    }


//--------allocation des tableaux sur le device---------
    int *dev_a, *dev_b, *dev_m;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, lenA * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, lenB * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_m, lenM * sizeof(int) ) );


    HANDLE_ERROR( cudaMemcpy( dev_a, A, lenA * sizeof(int), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, B, lenB * sizeof(int), cudaMemcpyHostToDevice ) );


//--------calcul du nombre de threads et nombre de blocs------------
    int blocksize = 1024;
    int nbBlock = 10* (lenM + blocksize -1)/blocksize;
    printf(" d=%d , threads= %d , blocs= %d\n",d,blocksize,nbBlock);

//------- calculs ---------
    float gpu_time;

    Timer timer = Timer();
    timer.start();

    printf("begining sorting\n");
    
    // merge_Small_k<<<1,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m);
    pathBig_k<<<nbBlock,blocksize>>>(dev_a,lenA,dev_b,lenB,dev_m);
    cudaDeviceSynchronize();
    printf("synchronized\n");

    timer.add();
    gpu_time = timer.getsum();

    // cudaMemoryTest();
    HANDLE_ERROR( cudaMemcpy( M, dev_m, lenM* sizeof(int), cudaMemcpyDeviceToHost ) );
    printf("memcopy M : done\n");

    affiche_tab(M,lenM);

    printf("gpu time: %f\n", gpu_time );
    

//------- liberation memoire ---------------

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_m);
    free(A);
    free(B);
    free(M);

};





/*##########################################################################
                                    MAIN
/###########################################################################*/


int main(){
	// test_batchMerge_deterministic(4,10000);
    test_PathMerge(100000);
    
    for(int i=1 ; i<30 ; i++){
        // test_PathMerge(pow(2,i));
        // test_batchMerge_rand(pow(2,i), 1000);
    }

    return 0;
}


