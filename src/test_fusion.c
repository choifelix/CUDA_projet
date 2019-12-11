#include "merge.h"
#include "affichage.h"
#include <time.h>
#include <math.h>


void test(int d){
	int lenM = d;
	int lenA = rand()%(lenM -1) + 1;
	int lenB = lenM - lenA;

	int *A;
	int *B;
	int *M;

	A = (int*)malloc(lenA*sizeof(int));
	B = (int*)malloc(lenB*sizeof(int));
	M = (int*)malloc(lenM*sizeof(int));

	// for (int i=0 ; i <lenA ;i++){
 //        A[i] = 2*i;
 //    }

 //    for (int i=0 ; i <lenB ;i++){
 //        B[i] = 2*i +1;
 //    }

	A[0] = rand()%10;
	B[0] = rand()%10;

	for (int i=1 ; i <lenA ;i++){
		A[i] = A[i-1] + rand()%10;
	}

	for (int i=1 ; i <lenB ;i++){
		B[i] = B[i-1] + rand()%10;
	}


	double cpu_time;

	long clk_tck = CLOCKS_PER_SEC;
   	clock_t t1, t2;

   	t1 = clock();
	merge( A , B, M,lenA,lenB);
	t2 = clock();

	cpu_time = (double)(t2-t1)/(double)clk_tck;

	// affiche_tab(M,lenM);

	printf("Temps (s) : %lf \n",cpu_time);

	free(A);
    free(B);
	free(M);
}


int main()
{
	for(int i=1 ; i<= 30 ; i++){
		test(pow(2,i));
	}
	return 0;
}