#include <stdio.h>
#include <stdlib.h>
#include "merge.h"


void merge( int * A , int * B, int * M,int len_A, int len_B )
{
	int i=0;
	int j=0; 

	while(i< len_A && j< len_B)
	{
		if (A[i] < B[j])
		{
			M[i+j]=A[i];
			i++;
		}
		else{
			M[i+j]=B[j];
			j++;
		}
	}
	while(i<len_A)     //lorsque len_A > Len_B  on remplit avec les derniers termes de A
	{
		M[i+j]=A[i];
		i++;

	}

	while(j<len_B)		//lorsque len_B > Len_A  on remplit avec les derniers termes de B
	{
		M[i+j]=B[j];
		j++;
	}

}


