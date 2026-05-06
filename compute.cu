#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

#define BLOCKSIZE 256


//kernel call to compute pairewise accelerations
__global__ void computePairWise(vector3* accels, vector3* pos, double* mass, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i>=N) return;

	__shared__ vector3 shPos[BLOCKSIZE];
	__shared__ double shMass[BLOCKSIZE];

	vector3 currPos;
	for(int t = 0; t < 3; t++){
		currPos[t]=pos[i][t];
	}//end to for

	for(int k = 0; k < N; k+=BLOCKSIZE){
		int j = k + threadIdx.x;
		if(j<N){
                        for(int t = 0; t < 3; t++){
                                shPos[threadIdx.x][t]=pos[j][t];
                        }//end to for
                        shMass[threadIdx.x]=mass[j];
                }//end to if

		__syncthreads();

		for(int m = 0; m < BLOCKSIZE; m++){
			int j = k + m;
			if(j < N && j !=i){
				vector3 dist;
                                for(int t = 0; t < 3; t++) {
                                        dist[t] = currPos[t] - shPos[m][t];
                                }//end to for
                                double mag_sq = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];

                                double mag = sqrt(mag_sq);
                                double accelMag = -1 * GRAV_CONSTANT * shMass[m] / mag;

				for(int t = 0; t< 3; t++){
					accels[i*N + j][t] = accelMag * dist[t]/mag_sq;
				}//end to for
			}//end to if
		}//end to for
	}//end to for

}//end to computePaireWise

//kernel call to sum up rows of mat
__global__ void sum(vector3* accels, vector3* accel_sum, int N){
	int i = blockIdx.x *  blockDim.x + threadIdx.x;

	if(i>=N) return;

	vector3 sum = {0,0,0};

	for(int j = 0; j < N; j++){
		for(int k = 0; k < 3; k++){
			sum[k]  += accels[i*N+j][k];
		}//end to inner for k
	}//end to outer for j

	for(int k = 0; k < 3; k++){
		accel_sum[i][k] = sum[k];
	}//end to for
}//end to sum

//kernel call to update new velocity and position
__global__ void updateVP(vector3* hVel, vector3* hPos, vector3* accel_sum, double interval, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i>=N) return;

	for(int k = 0; k < 3; k++){
		hVel[i][k] += accel_sum[i][k] * interval;
		hPos[i][k] += hVel[i][k]*interval; 
	}//end to for

}//end to updateVP

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	//int i,k;
	/*vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	*/
	vector3* d_accels;
        cudaMalloc((void**)&d_accels, sizeof(vector3)*NUMENTITIES);

	double* d_mass;
        cudaMalloc((void**)&d_mass, sizeof(double)*NUMENTITIES);

	vector3* d_pos;
	cudaMalloc((void**)&d_pos, sizeof(vector3)*NUMENTITIES);

        cudaMemcpy(d_pos, hPos, sizeof(vector3)* NUMENTITIES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCKSIZE);
        dim3 gridSize((NUMENTITIES + BLOCKSIZE-1)/BLOCKSIZE);

	computePairWise<<<gridSize, blockSize>>>(d_accels, d_pos, d_mass, NUMENTITIES);
        cudaDeviceSynchronize();

        vector3* accels = (vector3*)malloc(sizeof(vector3)*NUMENTITIES);
        cudaMemcpy(accels, d_accels, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
/*
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
*/

	vector3* d_accel_sum;
	cudaMalloc((void**)&d_accel_sum, sizeof(vector3)*NUMENTITIES);
	sum<<<gridSize, blockSize>>>(d_accels, d_accel_sum, NUMENTITIES);

	vector3* d_vel;
	cudaMalloc((void**)&d_vel, sizeof(vector3)*NUMENTITIES);
	cudaMemcpy(d_vel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	updateVP<<<gridSize, blockSize>>>(d_vel, d_pos, d_accel_sum, INTERVAL, NUMENTITIES);

	cudaMemcpy(hPos, d_pos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_vel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	/*for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (int j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}*/
	cudaFree(d_accels);
        cudaFree(d_mass);
	cudaFree(d_accel_sum);
	cudaFree(d_pos);
	cudaFree(d_vel);
        free(accels);
}
