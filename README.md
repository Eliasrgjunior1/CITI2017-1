# Algoritmo Backpropagation
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>

#define ENTRADA   2
#define OCULTA_1  2
#define SAIDA     1

#define AMOSTRA 4
#define n 0.5
#define E 0.0001

struct mlp{		 // Estrutura do Neurônio
	
	float X[ENTRADA+1];  				//Entradas da redes   Camada Entrada
	
	float W1[OCULTA_1][ENTRADA+1];  	//Pesos sinápticos 	  1ª Camada
	float dE_dW1[OCULTA_1][ENTRADA+1];	//Gradiente de erro   1ª Camada
	float I1[OCULTA_1];			 		//Somatório  		  1ª Camada
	float Y1[OCULTA_1+1];				//Saidas 			  1ª Camada
	
	float W2[SAIDA][OCULTA_1+1];  		//Pesos sinápticos 	  Camada Saida
	float dE_dW2[SAIDA][OCULTA_1+1];	//Gradiente de erro   Camada Saida
	float g2[SAIDA];					//Gradiente de erro   Camada Saida
	float I2[SAIDA];			 		//Somatório  		  Camada Saida
	float Y2[SAIDA];					//Saidas 			  Camada Saida
	
};

struct mlp net;

float X_train[AMOSTRA][ENTRADA]={

0,0,
0,1,
1,0,
1,1

};

float Y_train[AMOSTRA][SAIDA]={ 1,0,0,1  };
	
float X_teste[10][ENTRADA]={
	
0,0,
0,1,
1,0,
1,1

};


void nn_PESOS(){     //inicializa com valores aleatórios
	
	net.X [0]= 1;
	net.Y1[0]= 1;
	
	srand(time(0));
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  net.W1[j][i]=((rand()%19)*0.1)-0.9;
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) net.W2[j][i]=((rand()%19)*0.1)-0.9; 
		
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  printf("W%i%i: %f  \n",j,i,net.W1[j][i]);
	printf("\n");
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) printf("W%i%i: %f  \n",j,i,net.W2[j][i]);
	printf("\n\n\n");
	
}

float nn_F_ATIVA( int tipo, float soma){		// Função para escolha da função de ativação de cada camada
	
	float f_ativa;
	
	if(tipo)
		f_ativa=1/(1+(exp(-soma)) );
	
	return(f_ativa);
	
}

void nn_PROPAGAR(){	//propaga sinal pela rede
	
	
	for( int j=0; j<OCULTA_1 ;j++ ) net.I1[j] = 0;
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ ) net.I1[j] += net.W1[j][i] * net.X[i];	
	for( int j=1; j<=OCULTA_1 ;j++ ) net.Y1[j] = nn_F_ATIVA(1,net.I1[j-1]);

	for( int j=0; j<SAIDA ;j++ ) net.I2[j] = 0;
	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )   net.I2[j] += net.W2[j][i] * net.Y1[i];
	for( int j=0; j<SAIDA ;j++ ) net.Y2[j] = nn_F_ATIVA(1,net.I2[j]);
	
}


nn_PADRAO(int fonte, int amostra){		//Exibe padrao a rede (fonte de onde ele tira os padrões usuario ou treino  ////   amostra de treino a ser exibida )
	
	
	if(fonte){
		
		for(int i=1; i<=ENTRADA; i++) net.X[i]=X_train[amostra][i-1];	
		
	}
	else{
		
		for(int i=1; i<=ENTRADA; i++) {
			
			printf("Entre com x%i \n",i);
			scanf("%f",&net.X[i]);
			
		}
		
	}
	
	
}



void nn_RETROPROPAGAR(int amostra){
	
	for( int j=0; j<SAIDA ;j++ ) net.g2[j] = ( Y_train[amostra][j] - net.Y2[j] ) * ( net.Y2[j]*(1-net.Y2[j]) );	  //Calcula gradiente na primeira camada
	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )  net.dE_dW2[j][i] = -1 * net.g2[j] * net.Y1[i];  //Matriz dos gradientes 
	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )  net.W2[j][i] -= n*net.dE_dW2[j][i] ;			  //Atualização dos pesos


//SE der erro ele esta no gradiente	
	for( int j=0; j<OCULTA_1 ;j++ ) 
		for(int i=0; i<=ENTRADA; i++ ) {
		
			float gradiente=0;	
			for( int k=0; k<SAIDA ;k++ ) gradiente+= net.g2[k] * net.W2[k][j+1];
			net.dE_dW1[j][i] = -1 * gradiente * ( net.Y1[j+1]*(1-net.Y1[j+1]) ) * net.X[i] ;
			
		}
	
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )   net.W1[j][i] -= n*net.dE_dW1[j][i] ;	  //Atualização dos pesos		
	
}


float nn_EQM(){
	
	float eqm=0,erro;
	
	for(int a=0;a<AMOSTRA;a++){
		
		nn_PADRAO(1,a);
		nn_PROPAGAR();
		erro=0;
		for( int j=0; j<SAIDA ;j++ )
			erro += Y_train[a][j] - net.Y2[j];
			
		eqm  += (erro*erro);
	}
	
	eqm*=0.5;
	
	return(eqm/AMOSTRA);
	
}


void nn_TREINO(){			// Regra DELTA
	
	int epocas=0;
	float eqm_ant=0, eqm_atual=nn_EQM(), eqm_delta ;
	
	do{
	
		eqm_ant=eqm_atual;
		
		for(int a=0; a<AMOSTRA; a++ ){
			
			nn_PADRAO(1,a);
			
			nn_PROPAGAR();
				
			nn_RETROPROPAGAR(a);
			
		}
		
		eqm_atual=nn_EQM();
		
		eqm_delta = eqm_atual-eqm_ant;
		
		if(0>eqm_delta)eqm_delta*=-1;
	
		epocas++;
		
	}while(eqm_atual>=E && epocas<50000);
	
	printf("Treinado  I=%i    E=%.5f\n",epocas,eqm_atual);
	
	printf("\n\n\n");
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  printf("W%i%i: %f  \n",j,i,net.W1[j][i]);
	printf("\n");
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) printf("W%i%i: %f  \n",j,i,net.W2[j][i]);
	printf("***************\n\n\n");
	
}



main(){
	
	nn_PESOS();
		
	nn_TREINO();
	
	printf("\n\n");
	
/*	for(int a=0; a<10 ; a++){
		
		for(int i=1; i<=ENTRADA; i++) neuro.x[i]=X_teste[a][i-1];
		for(int i=1; i<=ENTRADA; i++) printf("x%i:  %f ",i,neuro.x[i]);
		nn_SINAL(1);
		printf("    Amostra %i:  %i \n",a,neuro.y);
		
		
	}
	
	printf("\n\n");
*/	
	for(;;){
		
		nn_PADRAO(0,0);
				
		nn_PROPAGAR();
		
		printf("Y:%i\n\n",0.5<net.Y2[0]);
	}
	
	
}

# Risilient Propagation
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>

#define ENTRADA   2
#define OCULTA_1  4
#define SAIDA     1

#define AMOSTRA 4
#define npos 1.2
#define nneg 0.5
#define MAX  50
#define MIN  0.000001

#define n 0.5
#define E 0.000001

struct mlp{		 // Estrutura do Neurônio
	
	float X[ENTRADA+1];  				//Entradas da redes   Camada Entrada

	
	float dE_dW1_ANT[OCULTA_1][ENTRADA+1];	//Gradiente de erro   1ª Camada
	float dW1[OCULTA_1][ENTRADA+1];  	//Pesos sinápticos 	  1ª Camada	


	float W1[OCULTA_1][ENTRADA+1];  	//Pesos sinápticos 	  1ª Camada
	float dE_dW1[OCULTA_1][ENTRADA+1];	//Gradiente de erro   1ª Camada
	float I1[OCULTA_1];			 		//Somatório  		  1ª Camada
	float Y1[OCULTA_1+1];				//Saidas 			  1ª Camada

	
	float dE_dW2_ANT[SAIDA][OCULTA_1+1];	//Gradiente de erro   Camada Saida
	float dW2[SAIDA][OCULTA_1+1];  			//Pesos sinápticos 	  Camada Saida

		
	float W2[SAIDA][OCULTA_1+1];  		//Pesos sinápticos 	  Camada Saida
	float dE_dW2[SAIDA][OCULTA_1+1];	//Gradiente de erro   Camada Saida
	float g2[SAIDA];					//Gradiente de erro   Camada Saida
	float I2[SAIDA];			 		//Somatório  		  Camada Saida
	float Y2[SAIDA];					//Saidas 			  Camada Saida
	
};

struct mlp net;

float X_train[AMOSTRA][ENTRADA]={

0,0,
0,1,
1,0,
1,1

};

float Y_train[AMOSTRA][SAIDA]={ 1,0,0,1  };
	

void nn_PESOS(){     //inicializa com valores aleatórios
	
	net.X [0]= 1;
	net.Y1[0]= 1;
	
	srand(time(0));
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  net.W1[j][i]=((rand()%19)*0.1)-0.9;
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) net.W2[j][i]=((rand()%19)*0.1)-0.9; 
	
	
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) net.dE_dW2_ANT[j][i]=0;
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )	net.dE_dW1_ANT[j][i]=0;
	
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) net.dW2[j][i]=0.1;
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )	net.dW1[j][i]=0.1;
	
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  printf("W%i%i: %f  \n",j,i,net.W1[j][i]);
	printf("\n");
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) printf("W%i%i: %f  \n",j,i,net.W2[j][i]);
	printf("\n\n\n");
	
}

float nn_F_ATIVA( int tipo, float soma){		// Função para escolha da função de ativação de cada camada
	
	float f_ativa;
	
	if(tipo)
		f_ativa=1/(1+(exp(-soma)) );
	
	return(f_ativa);
	
}

void nn_PROPAGAR(){	//propaga sinal pela rede
	
	
	for( int j=0; j<OCULTA_1 ;j++ ) net.I1[j] = 0;
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ ) net.I1[j] += net.W1[j][i] * net.X[i];	
	for( int j=1; j<=OCULTA_1 ;j++ ) net.Y1[j] = nn_F_ATIVA(1,net.I1[j-1]);

	for( int j=0; j<SAIDA ;j++ ) net.I2[j] = 0;
	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )   net.I2[j] += net.W2[j][i] * net.Y1[i];
	for( int j=0; j<SAIDA ;j++ ) net.Y2[j] = nn_F_ATIVA(1,net.I2[j]);
	
}


nn_PADRAO(int fonte, int amostra){		//Exibe padrao a rede (fonte de onde ele tira os padrões usuario ou treino  ////   amostra de treino a ser exibida )
	
	
	if(fonte){
		
		for(int i=1; i<=ENTRADA; i++) net.X[i]=X_train[amostra][i-1];	
		
	}
	else{
		
		for(int i=1; i<=ENTRADA; i++) {
			
			printf("Entre com x%i \n",i);
			scanf("%f",&net.X[i]);
			
		}
		
	}
	
	
}

void nn_ZERAR(){
	

	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )  net.dE_dW2[j][i] = 0;  //Matriz dos gradientes 

	for( int j=0; j<OCULTA_1 ;j++ )	for(int i=0; i<=ENTRADA; i++ )	net.dE_dW1[j][i] = 0;
	
}


void nn_JACOBIAN(int amostra){
	

	for( int j=0; j<SAIDA ;j++ ) net.g2[j] = ( Y_train[amostra][j] - net.Y2[j] ) * ( net.Y2[j]*(1-net.Y2[j]) );	  //Calcula gradiente na primeira camada
	for( int j=0; j<SAIDA ;j++ ) for(int i=0; i<=OCULTA_1; i++ )  net.dE_dW2[j][i] += -1 * net.g2[j] * net.Y1[i];  //Matriz dos gradientes 


	for( int j=0; j<OCULTA_1 ;j++ ) 
		for(int i=0; i<=ENTRADA; i++ ) {
		
			float gradiente=0;	
			for( int k=0; k<SAIDA ;k++ ) gradiente+= net.g2[k] * net.W2[k][j+1];
			net.dE_dW1[j][i] += -1 * gradiente * ( net.Y1[j+1]*(1-net.Y1[j+1]) ) * net.X[i] ;
			
		}

		
	
}

void nn_RETROPROPAGAR(){
	
	
	for( int j=0; j<SAIDA ;j++ )   for(int i=0; i<=OCULTA_1; i++ ) {
		
		float s=net.dE_dW2[j][i]*net.dE_dW2_ANT[j][i];
		
		if(s>0){
			
//			printf(" positivo\t");
			if(net.dW2[j][i]<MAX) net.dW2[j][i]*=npos;
			net.dE_dW2_ANT[j][i]=net.dE_dW2[j][i];
			
		}
		if(s<0){
	//			printf(" negativo \t");
			if(net.dW2[j][i]>MIN) net.dW2[j][i]*=nneg;
			net.dE_dW2_ANT[j][i]=0;
			
		}
		if(s==0){
	//		printf(" zero     \t");
			net.dE_dW2_ANT[j][i]=net.dE_dW2[j][i];
			
		}
		
		if(net.dE_dW2[j][i]>0)  net.W2[j][i] -= net.dW2[j][i];
	
		if(net.dE_dW2[j][i]<0)	net.W2[j][i] += net.dW2[j][i];
		
	}
/*
	printf("\n");
	for( int j=0; j<SAIDA ;j++ ) 
		for(int i=0; i<=OCULTA_1; i++ ){
			
			printf("%.6f\t",net.dW2[j][i]);
			
		}
		printf("\n");
		
*/
	
	for( int j=0; j<OCULTA_1 ;j++ ){
	
//	printf("\n");
	
	   for(int i=0; i<=ENTRADA; i++ )  {
		
		float s=net.dE_dW1[j][i]*net.dE_dW1_ANT[j][i];
		
		if(s>0){
			
	//		printf(" positivo\t");
			if(net.dW1[j][i]<MAX)  net.dW1[j][i]*=npos;
			net.dE_dW1_ANT[j][i]=net.dE_dW1[j][i];
			
		}
		
		if(s<0){
			
		//	printf(" negativo \t");
			if(net.dW1[j][i]>MIN)  net.dW1[j][i]*=nneg;
			net.dE_dW1_ANT[j][i]=0;
			
		}
		
		if(s==0){
		//	printf(" zero     \t");
			net.dE_dW1_ANT[j][i]=net.dE_dW1[j][i];
			
		}
		
		if(net.dE_dW1[j][i]>0)  net.W1[j][i] -= net.dW1[j][i];
		
		if(net.dE_dW1[j][i]<0)	net.W1[j][i] += net.dW1[j][i];
		
		}
	}
/*	
	printf("\n");
	for( int j=0; j<OCULTA_1 ;j++ ){ 
		for(int i=0; i<=ENTRADA; i++ ){
			
			printf("%.6f\t",net.dW1[j][i]);
			
		}
		printf("\n");
	}
	printf("\n");*/
	
//	for( int j=0; j<SAIDA ;j++ )   for(int i=0; i<=OCULTA_1; i++ ) 	 net.W2[j][i] -= n*net.dE_dW2[j][i] ;	  //Atualização dos pesos	
	
//	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )   net.W1[j][i] -= n*net.dE_dW1[j][i] ;	  //Atualização dos pesos		
	
	
}


float nn_EQM(){
	
	float eqm=0,erro;
	
	for(int a=0;a<AMOSTRA;a++){
		
		nn_PADRAO(1,a);
		nn_PROPAGAR();
		erro=0;
		for( int j=0; j<SAIDA ;j++ )
			erro += Y_train[a][j] - net.Y2[j];
			
		eqm  += (erro*erro);
	}
	
	eqm*=0.5;
	
	return(eqm/AMOSTRA);
	
}


void Exibe_matriz(){
	
	for( int j=0; j<SAIDA ;j++ ){ 
		for(int i=0; i<=OCULTA_1; i++ ){
			
			printf("%.4f\t",net.dE_dW2[j][i]);
			
		}
		printf("\n");
	}
	
	printf("*****************************************\n");
	
	for( int j=0; j<OCULTA_1 ;j++ ) {
		for(int i=0; i<=ENTRADA; i++ ) {
			
			printf("%.4f\t",net.dE_dW1[j][i]);
			
		}
		printf("\n");
	}
	
	printf("*****************************************\n\n\n");
	//getch();
	
}

void nn_TREINO(){			// Regra DELTA
	
	int epocas=0;
	float eqm_ant=0, eqm_atual=nn_EQM(), eqm_delta ;
		
	do{
	
		eqm_ant=eqm_atual;
	
		nn_ZERAR();
	
		for(int a=0; a<AMOSTRA; a++ ){
			
			
			nn_PADRAO(1,a);
			
			nn_PROPAGAR();
			
			nn_JACOBIAN(a);
			
			
		}
		
		//Exibe_matriz();		
		
		nn_RETROPROPAGAR();
		
		
		eqm_atual=nn_EQM();
		
		eqm_delta = eqm_atual-eqm_ant;
		
		if(0>eqm_delta) eqm_delta*=-1;
		
		//if(0.0000000001>eqm_delta) nn_PESOS();
	
		epocas++;
		
		printf("Treinado  I=%i    E=%.10f	E=%.10f\n",epocas,eqm_atual,eqm_delta);	
		
	}while(eqm_atual>=E && epocas<4000);
	
	printf("Treinado  I=%i    E=%.8f\n",epocas,eqm_atual);
	
	printf("\n\n\n");
	for( int j=0; j<OCULTA_1 ;j++ ) for(int i=0; i<=ENTRADA; i++ )  printf("W%i%i: %f  \n",j,i,net.W1[j][i]);
	printf("\n");
	for( int j=0; j<SAIDA ;j++ ) 	for(int i=0; i<=OCULTA_1; i++ ) printf("W%i%i: %f  \n",j,i,net.W2[j][i]);
	printf("***************\n\n\n");
	
}



main(){
	
	nn_PESOS();
		
	nn_TREINO();
	
	printf("\n\n");
	
	
	for(;;){
		
		nn_PADRAO(0,0);
				
		nn_PROPAGAR();
		
		printf("Y:%i\n\n",0.5<net.Y2[0]);
		
	}
	
	
}

# Interpolação linear 2
#include <stdio.h>
#include <stdlib.h>

float linear_inter( float x1, float y1, float x2, float y2, float c  ){


    float  interpol ;
    
    interpol = ( (c-x1)/(x2-x1) )*(y2-y1) + y1;    
    
    return(interpol);
    
}

/*

main(){
    
    float x1,y1,x2,y2,c ;
    
    printf("entre com X1 e Y2\n");
    scanf("%f %f",&x1,&y1);
    
    printf("entre com X1 e Y2\n");
    scanf("%f %f",&x2,&y2);
    
    printf("entre X\n");
    scanf("%f",&c);
    
    
    printf("A Saida e: %f\n", linear_inter( x1,y1,x2,y2,c )); 
        
    system("pause");
    
}

*/

main(){
	
	
	FILE *arq;
	arq = fopen("Temp2.txt", "r");
	FILE *arq2;
	arq2 = fopen("Saida.txt", "w");	
	if(arq == NULL)	printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		
		float x1,y1,x2,y2;
		fscanf(arq,"%f %f",&x1,&y1);
		
		for( int i=1 ;feof(arq)==NULL; i++){

			fscanf(arq,"%f %f",&x2,&y2);			
			printf("%d: %f , %f \n",i,x1,y1);
			for(int j=1;j<10;j++) printf("      %d: %f , %f \n",j,x1+j/2.0, linear_inter(x1,y1,x2,y2,x1+j/2.0) );
			printf("%d: %f , %f \n",i,x2,y2);
			x1=x2;
			y1=y2;
		}
	
	}
	
	fclose(arq);
	fclose(arq2);
	
}

# Interpolação linear
#include <stdio.h>
#include <stdlib.h>

float linear_inter( float x1, float y1, float x2, float y2, float c  ){


    float  interpol ;
    
    interpol = ( (c-x1)/(x2-x1) )*(y2-y1) + y1;    
    
    return(interpol);
    
}

main(){
	
	
	FILE *arq;
	arq = fopen("Temperatura.txt", "r");
	FILE *arq2;
	arq2 = fopen("Teste.txt", "w");	
	if(arq == NULL)	printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		
		float x1,y1,x2,y2;
		
		//for( int i=1 ;feof(arq)==NULL; i++){

			//fscanf(arq,"%f",&x1);			
			//if(i%13==1) fprintf(arq2,"%f ",x1);
			//if(i%13==0) fprintf(arq2,"%f ",x1);
			for(int i=0;i<374*2+1;i++) fprintf(arq2,"%f ",i/2.0);
		//}
	
	}
	
	fclose(arq);
	fclose(arq2);
	
}

# Interpolação quadrática 2
#include <stdio.h>
#include <stdlib.h>

double a,b,c;

void quad_inter(float x1, float y1, float x2, float y2, float x3, float y3){
	
	
	double entrada[3][3]={ x1*x1, x1, 1 , x2*x2, x2, 1 , x3*x3, x3, 1  }, saida[3]={y1,y2,y3};

	float esc=0;
	
	for(int x=1;x<3;x++){
	
	
		esc =  -entrada[x][0]/entrada[0][0];
		
		for(int y=0;y<3;y++) entrada[x][y]+= esc*entrada[0][y];
		
		saida[x]+=esc*saida[0];
	
	}

	esc =  -entrada[2][1]/entrada[1][1];
		
	for(int y=1;y<3;y++) entrada[2][y]+= esc*entrada[1][y];
		
	saida[2]+=esc*saida[1];
	

	c = saida[2]/entrada[2][2];	
	b = (saida[1] - entrada[1][2]*c )/entrada[1][1];	
	a = (saida[0] - entrada[0][1]*b - entrada[0][2]*c )/entrada[0][0];
		
}

main(){
	
	
	FILE *arq;
	arq = fopen("Temp2.txt", "r");
	FILE *arq2;
	arq2 = fopen("Saida.txt", "w");	
	if(arq == NULL)	printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		
		float x1,y1,x2,y2,x3,y3;
		fscanf(arq,"%f %f",&x1,&y1);
		fscanf(arq,"%f %f",&x2,&y2);
		
		for( int i=1 ;feof(arq)==NULL; i++){

			fscanf(arq,"%f %f",&x3,&y3);			
			printf("%d: %f , %f \n",i,x1,y1);			
			printf("%d: %f , %f \n",i,x2,y2);
			printf("%d: %f , %f \n",i,x3,y3);
			printf("  \n");
			
			quad_inter(x1,y1,x2,y2,x3,y3);			
			
			for(int j=1; j<10 ;j++ ){
				
				printf("     %d:  %f  %f\n",j, x1+j/2.0, a*(x1+j/2.0)*(x1+j/2.0)+b*(x1+j/2.0)+c   );
				
			}
			
			x1=x2;
			y1=y2;
			x2=x3;
			y2=y3;
			
		}
	
	}
	
	fclose(arq);
	fclose(arq2);
	
}

# Interpolação quadrática
#include <stdio.h>
#include <stdlib.h>

double a,b,c;

void quad_inter(float x1, float y1, float x2, float y2, float x3, float y3){
	
	
	double entrada[3][3]={ x1*x1, x1, 1 , x2*x2, x2, 1 , x3*x3, x3, 1  }, saida[3]={y1,y2,y3};
	
/*
	for(int x=0;x<3;x++){
	
		printf(" %.2fa  +  (%.2f)b  +  (%.2f)c  =  %f \n",entrada[x][0],entrada[x][1],entrada[x][2],saida[x]);
		
	}
	
*/	
	float esc=0;
	
	for(int x=1;x<3;x++){
	
	
		esc =  -entrada[x][0]/entrada[0][0];
	//	printf(" \n%f\n",esc);
		
		for(int y=0;y<3;y++) entrada[x][y]+= esc*entrada[0][y];
		
		saida[x]+=esc*saida[0];
	
	}
/*	
	for(int x=0;x<3;x++){
	
		printf(" %.2fa  +  (%.2f)b  +  (%.2f)c  =  %f \n",entrada[x][0],entrada[x][1],entrada[x][2],saida[x]);
		
	}
	
*/	
	esc =  -entrada[2][1]/entrada[1][1];
//	printf(" \n%f\n",esc);
		
	for(int y=1;y<3;y++) entrada[2][y]+= esc*entrada[1][y];
		
	saida[2]+=esc*saida[1];
	
/*	
	for(int x=0;x<3;x++){
	
		printf(" %.2fa  +  (%.2f)b  +  (%.2f)c  =  %f \n",entrada[x][0],entrada[x][1],entrada[x][2],saida[x]);
		
	}
*/	
	
	
	c = saida[2]/entrada[2][2];
	
	b = (saida[1] - entrada[1][2]*c )/entrada[1][1];
	
//	printf("%f",entrada[1][2]);
	
	a = (saida[0] - entrada[0][1]*b - entrada[0][2]*c )/entrada[0][0];
	
//	printf("\n\n %.8f x2 + %.2f x + %.2f  = y",a,b,c );
	
}

main(){
	
	
	FILE *arq;
	arq = fopen("Temp2.txt", "r");
	FILE *arq2;
	arq2 = fopen("Saida.txt", "w");	
	if(arq == NULL)	printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		
		float x1,y1,x2,y2,x3,y3;
		fscanf(arq,"%f %f",&x1,&y1);
		fscanf(arq,"%f %f",&x2,&y2);
		
		for( int i=1 ;feof(arq)==NULL; i++){

			fscanf(arq,"%f %f",&x3,&y3);			
			printf("%d: %f , %f \n",i,x1,y1);			
			printf("%d: %f , %f \n",i,x2,y2);
			printf("%d: %f , %f \n",i,x3,y3);
			printf("  \n");
			
			quad_inter(x1,y1,x2,y2,x3,y3);			
			
			for(int j=1; j<10 ;j++ ){
				
				printf("     %d:  %f  %f\n",j, x1+j/2.0, a*(x1+j/2.0)*(x1+j/2.0)+b*(x1+j/2.0)+c   );
				
			}
			
			x1=x2;
			y1=y2;
			x2=x3;
			y2=y3;
			
		}
	
	}
	
	fclose(arq);
	fclose(arq2);
	
}








