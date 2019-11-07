#include "affichage.h"

void affiche_tab(int * Tab, int len_tab)
{
	for(int i=0; i < len_tab; i++)
	{
		printf("%d\t",Tab[i]);
	}
}