#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void computeAccelMat(vector3* accels, vector3* pos, double* mass, int N) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < N && j < N){
		int ind = i*N+j;

		if(i==j){
			accels[ind][0]=0;
			accels[ind][1]=0;
			accels[ind][2]=0;
		}
		else{
			vector3 dist;
			for(int k = 0; k<3; k++){
				dist[k] = pos[i][k] - pos[j][k];
			}//end to for
			double mag_sq = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];
			double mag = sqrt(mag_sq);
			double accelMag = -1 * GRAV_CONSTANT * mass[j] / mag_sq;

			accels[ind][0] = accelMag * dist[0] / mag;
			accels[ind][1] = accelMag * dist[1] / mag;
			accels[ind][2] = accelMag * dist[2] / mag;
		}
	}//end to if
}//end to compute accel matrix


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	/*vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	*/
	vector3* d_accels;
	cudaMalloc((void**)&d_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	
	double* d_mass;
	cudaMalloc((void**)&d_mass, sizeof(double)*NUMENTITIES);

	cudaMemcpy(d_hPos, hPos, sizeof(vector3)* NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	//first compute the pairwise accelerations.  Effect is on the first argument.
	/*for (i=0;i<NUMENTITIES;i++){
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
	}*/

	dim3 blockSize(16,16);
	dim3 gridSize((NUMENTITIES + 15) / 16,(NUMENTITIES + 15)/16);

	computeAccelMat<<<gridSize, blockSize>>>(d_accels, d_hPos, d_mass, NUMENTITIES);
	cudaDeviceSynchronize();

	vector3* accels = (vector3*)malloc(sizeof(vector3) *NUMENTITIES*NUMENTITIES);
	cudaMemcpy(accels, d_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyDeviceToHost);
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i*NUMENTITIES+j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	cudaFree(d_accels);
	cudaFree(d_mass);
	free(accels);
	//free(values);
}
