#include <stdio.h>
#include <stdlib.h>
#include <time.h>


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


            	


int main()
{
	int d=8;
	int N=8;

	int *A[N];
    int *B[N];
    int lenA[N];
    int lenB[N];

    construct_input(A,lenA,B,lenB,d,N);
    printf("Ma liste A\n");
    affiche_list(A,lenA,N);
    printf("Ma liste B :\n");
    affiche_list(B,lenB,N);


}