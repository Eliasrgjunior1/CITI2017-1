# CITI2017-1
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

