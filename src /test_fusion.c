#include "merge.h"
#include "affichage.h"




int main()
{
	int B[6]={3,5,7,9,12,13};
	int A[4]={1,6,10,11};
	int len_A=4;
	int len_B=6;
	int *M;
	M=(int*)malloc((len_A + len_B)*sizeof(int));

	merge( A , B, M,len_A,len_B);
	affiche_tab(M,10);
	free(M);

}